#!/usr/bin/env bash
# Create the campaign machine: one instance template, one managed instance
# group (MIG) of size 1.
#
# The MIG is not bureaucracy. A spot VM that GCP reclaims is deleted, and a
# managed group is the only GCP mechanism that recreates it without a human
# at the keyboard. Everything else here follows from that one requirement:
# the termination action must be DELETE (a STOPped instance is not recreated
# by a MIG, it just sits there), the code version must be pinned in metadata
# (a VM recreated next week must not run code written after the
# pre-registration), and the results must live in GCS (a recreated VM has an
# empty disk).
#
# Every run so far was a hand-built VM whose logs died with it. This script
# and cloud/sync.sh are the fix.
#
# Usage:
#     cloud/provision.sh                                  # defaults below
#     PQL_BLOCK=pq_n16 PQL_WORKERS=96 cloud/provision.sh
#     cloud/campaign.sh up                                # same, with guards
#
# Takes about 60 s. The first boot is asynchronous: the machine needs ~10 min
# to clone, build the venv, pass test_pq.py and restore the GCS state before
# the first job starts. Watch it with `cloud/campaign.sh status`.
set -euo pipefail

PQL_PROJECT="${PQL_PROJECT:-project-b2dccd70-9346-4730-9ef}"
PQL_ZONE="${PQL_ZONE:-northamerica-northeast1-b}"
PQL_MACHINE_TYPE="${PQL_MACHINE_TYPE:-n2-standard-96}"
PQL_DISK_SIZE="${PQL_DISK_SIZE:-200GB}"
PQL_DISK_TYPE="${PQL_DISK_TYPE:-pd-balanced}"
# Debian 13 (trixie) for its Python 3.13. Not a preference: requirements.txt
# pins scipy 1.18.0, which needs Python >= 3.12, and Debian 12 ships 3.11 --
# the boot dies at pip with "No matching distribution found". Loosening the
# pin instead would be worse: the committed archives were produced under these
# exact versions, and the converged grid has to nest bit-exactly onto them at
# steps 1..400.
PQL_IMAGE_FAMILY="${PQL_IMAGE_FAMILY:-debian-13}"
PQL_IMAGE_PROJECT="${PQL_IMAGE_PROJECT:-debian-cloud}"
# Kept in the template hash below, and deliberately unused: --max-run-duration
# needs a termination action of DELETE, which a MIG rejects. See the notes at
# the end of this file.
PQL_MAX_RUN_DURATION="${PQL_MAX_RUN_DURATION:-none}"
PQL_BUCKET="${PQL_BUCKET:-gs://${PQL_PROJECT}-pql-campaign}"
PQL_PREFIX="${PQL_PREFIX:-converged}"
PQL_BLOCK="${PQL_BLOCK:-converged_pilot}"
PQL_SIZES="${PQL_SIZES:-}"
PQL_INSTANCES="${PQL_INSTANCES:-}"
PQL_WORKERS="${PQL_WORKERS:-96}"
PQL_JOBS="${PQL_JOBS:-1}"
PQL_SYNC_INTERVAL="${PQL_SYNC_INTERVAL:-120}"
PQL_RESTART_CACHE="${PQL_RESTART_CACHE:-results/restart_cache}"
PQL_LANDSCAPE_URL="${PQL_LANDSCAPE_URL:-https://github.com/Ilyes-Jamoussi/peaked-circuit-landscape.git}"
PQL_REPRO_URL="${PQL_REPRO_URL:-https://github.com/Ilyes-Jamoussi/peaked-circuits-pennylane.git}"
PQL_LANDSCAPE_COMMIT="${PQL_LANDSCAPE_COMMIT:-}"   # default: local HEAD
PQL_REPRO_COMMIT="${PQL_REPRO_COMMIT:-}"           # default: requirements.txt
PQL_NAME_PREFIX="${PQL_NAME_PREFIX:-pql}"
PQL_SKIP_QUOTA_CHECK="${PQL_SKIP_QUOTA_CHECK:-0}"

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$CLOUD_DIR")"
REGION="${PQL_ZONE%-*}"

die() { printf 'provision: %s\n' "$*" >&2; exit 1; }
note() { printf 'provision: %s\n' "$*"; }

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | cut -d' ' -f1
    else
        shasum -a 256 | cut -d' ' -f1
    fi
}

command -v gcloud >/dev/null 2>&1 || die "gcloud not on PATH"
gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null \
    | grep -q . || die "no active gcloud account; run 'gcloud auth login'"

for script in startup.sh sync.sh shutdown.sh; do
    [ -f "$CLOUD_DIR/$script" ] || die "missing cloud/$script"
done

# The pinned commits. A recreated VM re-reads these from metadata, so they
# have to name code that exists on the remote, not just on this laptop.
if [ -z "$PQL_LANDSCAPE_COMMIT" ]; then
    PQL_LANDSCAPE_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
fi
if [ -z "$PQL_REPRO_COMMIT" ]; then
    PQL_REPRO_COMMIT="$(sed -n 's/.*checkout \([0-9a-f]\{7,40\}\).*/\1/p' \
        "$REPO_DIR/requirements.txt" | head -1)"
fi
[ -n "$PQL_REPRO_COMMIT" ] || die "no reproduction-repo commit found in requirements.txt; set PQL_REPRO_COMMIT"

git -C "$REPO_DIR" fetch --quiet origin >/dev/null 2>&1 || true
if ! git -C "$REPO_DIR" branch -r --contains "$PQL_LANDSCAPE_COMMIT" 2>/dev/null | grep -q .; then
    die "commit $PQL_LANDSCAPE_COMMIT is not on any remote branch: push it first,
     or the recreated VM will clone code that does not contain it"
fi

# Only startup.sh and shutdown.sh travel as metadata. Everything else the
# machine runs is read out of the clone at the pinned commit, so a pin that
# predates one of these files boots, logs "sync loop started", and mirrors
# nothing -- which is the exact failure this directory exists to prevent.
for path in cloud/sync.sh cloud/bootstrap.sh cloud/runner.py; do
    git -C "$REPO_DIR" cat-file -e "$PQL_LANDSCAPE_COMMIT:$path" 2>/dev/null \
        || die "commit $PQL_LANDSCAPE_COMMIT does not contain $path, and the VM
     runs that file from the clone, not from metadata. Commit and push it, or
     pin a commit that has it."
done

BLOCK_SLUG="$(printf '%s' "$PQL_BLOCK" | tr 'A-Z' 'a-z' | tr '_' '-' | tr -cd 'a-z0-9-')"
[ -n "$BLOCK_SLUG" ] || die "PQL_BLOCK '$PQL_BLOCK' has no usable name characters"
MIG_NAME="${PQL_NAME_PREFIX}-${BLOCK_SLUG}"

# The template name carries a hash of everything baked into it. Templates are
# immutable, so a changed pin or shard silently needs a new template; deriving
# the name from the configuration makes that automatic instead of a trap.
CONFIG_TAG="$(printf '%s\n' \
    "$PQL_LANDSCAPE_COMMIT $PQL_REPRO_COMMIT $PQL_BLOCK $PQL_SIZES $PQL_INSTANCES" \
    "$PQL_WORKERS $PQL_JOBS $PQL_BUCKET $PQL_PREFIX $PQL_MACHINE_TYPE $PQL_MAX_RUN_DURATION" \
    | sha256_of | cut -c1-8)"
TEMPLATE_NAME="${PQL_NAME_PREFIX}-${BLOCK_SLUG}-${CONFIG_TAG}"

note "project        $PQL_PROJECT"
note "zone           $PQL_ZONE ($PQL_MACHINE_TYPE, spot)"
note "block          $PQL_BLOCK  sizes='${PQL_SIZES:-all}' instances='${PQL_INSTANCES:-all}' workers=$PQL_WORKERS jobs=$PQL_JOBS"
note "landscape      $PQL_LANDSCAPE_COMMIT"
note "reproduction   $PQL_REPRO_COMMIT"
note "state          $PQL_BUCKET/$PQL_PREFIX"
note "template       $TEMPLATE_NAME"
note "group          $MIG_NAME"

if gcloud compute instance-groups managed describe "$MIG_NAME" \
        --project="$PQL_PROJECT" --zone="$PQL_ZONE" >/dev/null 2>&1; then
    current="$(gcloud compute instance-groups managed describe "$MIG_NAME" \
        --project="$PQL_PROJECT" --zone="$PQL_ZONE" \
        --format='value(instanceTemplate)')"
    if [ "$(basename "$current")" = "$TEMPLATE_NAME" ]; then
        note "group already runs this exact configuration; nothing to do"
        exit 0
    fi
    die "group $MIG_NAME exists on template $(basename "$current"), not
     $TEMPLATE_NAME. Fetch its results and run 'cloud/campaign.sh down'
     before provisioning a different configuration."
fi

if ! gcloud storage ls "$PQL_BUCKET" >/dev/null 2>&1; then
    note "creating $PQL_BUCKET in $REGION"
    gcloud storage buckets create "$PQL_BUCKET" \
        --project="$PQL_PROJECT" --location="$REGION" \
        --uniform-bucket-level-access
fi

# ------------------------------------------------------- identity preflight
# The VM does not run as the operator; it runs as the project's default
# compute service account, and that account's permissions are invisible from
# here unless they are checked on purpose. Everything above this point --
# quota, pinned commits, the presence of sync.sh in the pin -- was checked and
# passed while the machine's identity could read and write nothing at all. The
# campaign booted, ran, and every sync pass failed with 403 into a serial
# console nobody was reading.
#
# Two roles are needed and they fail differently, which is why both are
# checked:
#
#   storage.objectAdmin   the results. Without it the run produces nothing
#                         that outlives the instance, silently.
#   compute.instanceAdmin.v1
#                         the teardown. The finisher resizes its own managed
#                         group to 0 when the campaign ends; without this it
#                         reports TEARDOWN=denied and a 96-vCPU spot machine
#                         keeps billing after a run that SUCCEEDED.
#
# cloud/IAM.md carries the two grants. A policy this script cannot read is a
# warning rather than a stop: an operator may legitimately be allowed to use
# an identity whose policy they may not inspect.
PROJECT_NUMBER="$(gcloud projects describe "$PQL_PROJECT" \
    --format='value(projectNumber)' 2>/dev/null || true)"
if [ -n "$PROJECT_NUMBER" ]; then
    VM_SA="${PQL_SERVICE_ACCOUNT:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"
    note "vm identity    $VM_SA"

    # --flatten gives one `role<TAB>members` line per binding, and the match
    # is made here rather than with --filter: `gcloud storage buckets
    # get-iam-policy` does not accept --filter, and rejects it by exiting
    # non-zero -- which this block reads as an unreadable policy and skips.
    # A preflight that always skips is worse than none, because it prints a
    # line that looks like it checked something.
    unreadable=0
    bucket_policy="$(gcloud storage buckets get-iam-policy "$PQL_BUCKET" \
        --project="$PQL_PROJECT" --flatten='bindings[]' \
        --format='value(bindings.role,bindings.members)' 2>/dev/null)" \
        || unreadable=1
    project_policy="$(gcloud projects get-iam-policy "$PQL_PROJECT" \
        --flatten='bindings[]' \
        --format='value(bindings.role,bindings.members)' 2>/dev/null)" \
        || unreadable=1
    granted="$({ printf '%s\n%s\n' "$bucket_policy" "$project_policy" \
        | grep -F "$VM_SA" || true; } | cut -f1 | tr '\n' ' ')"

    case "$unreadable" in
        1)
            note "WARNING: cannot read the IAM policy; identity preflight skipped" ;;
        *)
            lacks=""
            case "$granted" in
                *storage.objectAdmin*|*storage.admin*|*roles/owner*|*roles/editor*) ;;
                *) lacks="$lacks write access to $PQL_BUCKET;" ;;
            esac
            case "$granted" in
                *compute.instanceAdmin*|*roles/owner*|*roles/editor*) ;;
                *) lacks="$lacks permission to resize its own instance group;" ;;
            esac
            [ -z "$lacks" ] || die "the VM's service account cannot do its job.
     $VM_SA lacks:$lacks
     so this campaign would either write nothing that outlives the machine, or
     fail to take that machine down when it finishes. The two grants are in
     cloud/IAM.md. Fix them, then run this again."
            note "iam            identity can write results and resize its group"
            ;;
    esac
fi

# Quota preflight, against BOTH ceilings. The regional CPUS limit (200 here)
# is the visible one, but a project also carries a global CPUS_ALL_REGIONS
# quota, and on this project it is 96 -- exactly one campaign machine. The
# first sharded launch passed the regional check, created its second group,
# and that group then failed every instance create with QUOTA_EXCEEDED against
# the global limit, invisibly to anyone who only read the regional number.
# Two 96-vCPU machines cannot coexist under a 96-CPU global cap, and no retry
# fixes that; the group just spins until the other machine frees the quota.
if [ "$PQL_SKIP_QUOTA_CHECK" != "1" ]; then
    needed="${PQL_MACHINE_TYPE##*-}"
    read -r _ used limit <<<"$(gcloud compute regions describe "$REGION" \
        --project="$PQL_PROJECT" --flatten='quotas[]' \
        --format='value(quotas.metric,quotas.usage,quotas.limit)' \
        | awk '$1 == "CPUS"')" || true
    if [ -n "${limit:-}" ]; then
        note "regional CPUS quota: ${used%.*}/${limit%.*} in use, this group needs $needed (x2 during a recreate)"
        if awk -v u="${used:-0}" -v l="${limit:-0}" -v n="${needed:-0}" \
               'BEGIN { exit !(u + n > l) }'; then
            die "not enough CPUS quota in $REGION for one more $PQL_MACHINE_TYPE.
     Delete the idle groups first, or set PQL_SKIP_QUOTA_CHECK=1 to let the
     MIG retry on its own."
        fi
    fi
    read -r _ gused glimit <<<"$(gcloud compute project-info describe \
        --project="$PQL_PROJECT" --flatten='quotas[]' \
        --format='value(quotas.metric,quotas.usage,quotas.limit)' 2>/dev/null \
        | awk '$1 == "CPUS_ALL_REGIONS"')" || true
    if [ -n "${glimit:-}" ]; then
        note "global CPUS_ALL_REGIONS quota: ${gused%.*}/${glimit%.*} in use, this group needs $needed"
        if awk -v u="${gused:-0}" -v l="${glimit:-0}" -v n="${needed:-0}" \
               'BEGIN { exit !(u + n > l) }'; then
            die "the global CPUS_ALL_REGIONS quota (${glimit%.*}) cannot hold one more
     $PQL_MACHINE_TYPE on top of the ${gused%.*} in use. The group would be
     created and then fail every instance create with QUOTA_EXCEEDED until
     something else frees the quota. Request an increase (IAM & Admin ->
     Quotas -> 'CPUs (all regions)'), wait for the running campaign to end,
     or set PQL_SKIP_QUOTA_CHECK=1 to queue behind it on purpose."
        fi
    fi
fi

if ! gcloud compute instance-templates describe "$TEMPLATE_NAME" \
        --project="$PQL_PROJECT" >/dev/null 2>&1; then
    note "creating instance template"
    gcloud compute instance-templates create "$TEMPLATE_NAME" \
        --project="$PQL_PROJECT" \
        --machine-type="$PQL_MACHINE_TYPE" \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --boot-disk-size="$PQL_DISK_SIZE" \
        --boot-disk-type="$PQL_DISK_TYPE" \
        --image-family="$PQL_IMAGE_FAMILY" \
        --image-project="$PQL_IMAGE_PROJECT" \
        --scopes=cloud-platform \
        --no-restart-on-failure \
        --labels="campaign=pql,block=$BLOCK_SLUG" \
        --metadata-from-file="startup-script=$CLOUD_DIR/startup.sh,shutdown-script=$CLOUD_DIR/shutdown.sh" \
        --metadata="^~^pql-landscape-url=$PQL_LANDSCAPE_URL~pql-landscape-commit=$PQL_LANDSCAPE_COMMIT~pql-repro-url=$PQL_REPRO_URL~pql-repro-commit=$PQL_REPRO_COMMIT~pql-bucket=$PQL_BUCKET~pql-prefix=$PQL_PREFIX~pql-block=$PQL_BLOCK~pql-sizes=$PQL_SIZES~pql-instances=$PQL_INSTANCES~pql-workers=$PQL_WORKERS~pql-jobs=$PQL_JOBS~pql-sync-interval=$PQL_SYNC_INTERVAL~pql-restart-cache=$PQL_RESTART_CACHE"
else
    note "instance template already exists (same configuration)"
fi

note "creating managed instance group"
gcloud compute instance-groups managed create "$MIG_NAME" \
    --project="$PQL_PROJECT" \
    --zone="$PQL_ZONE" \
    --template="$TEMPLATE_NAME" \
    --size=1 \
    --base-instance-name="${PQL_NAME_PREFIX}-${BLOCK_SLUG}"

note "done. Boot takes ~10 min; follow it with:"
note "    cloud/campaign.sh status"
note "    cloud/campaign.sh logs -f"

# ---------------------------------------------------------------------------
# Operating notes, all learned the hard way.
#
# Termination action. STOP, not DELETE, and this was learned from the API
# rather than from the documentation: a template carrying
# --instance-termination-action=DELETE is accepted, and then the group create
# fails with "Spot virtual machines with termination action set to DELETE
# cannot be used with Managed Instance Groups". DELETE is legal only for a
# standalone spot VM, which is precisely the case with no automatic recreate.
#
# So the recovery path is the MIG's own repair on a STOPped member, not a
# delete-and-rebuild. That is the better arrangement anyway: the boot disk
# survives a preemption, so the restart cache under results/ comes back with
# the machine and only the restarts that were actually in flight are lost.
# startup.sh is idempotent and restores from the bucket regardless, so the
# design does not depend on which of the two happens -- but a preempted
# machine now resumes from local state instead of re-downloading it.
#
# One consequence worth stating: a member that GCP cannot restart for lack of
# capacity stays STOPped, and the campaign waits rather than failing. That is
# the intended behaviour on spot, and `campaign.sh status` reports the group
# short of its target size while it lasts.
#
# Quota. The regional limit is CPUS = 200 (PREEMPTIBLE_CPUS is 0, and spot
# instances are charged against CPUS when it is, which is why an
# n2-highcpu-96 spot VM starts at all here). One group at 96 vCPU leaves 104
# free, so a recreate that overlaps the deletion of the old instance (192)
# still fits. Two groups do not: 192 steady-state plus one 96-vCPU overlap is
# 288, the create fails with QUOTA_EXCEEDED, and the MIG retries with
# backoff -- it recovers, but the machine can idle for several minutes and
# `campaign.sh status` will show the group short of its target size with a
# quota error in its last actions. Shard across sizes on ONE machine rather
# than running two groups, unless the quota has been raised.
#
# Machine family. n2-standard-96 is Intel. The archives already committed were
# produced on Intel and the converged grid must nest bit-exactly onto them at
# steps 1..400; an AMD (n2d, c3d) host changes the floating-point results and
# breaks that nesting. Do not "just take whatever spot capacity is available".
#
# Memory, not cores, sizes this machine. One n = 16 restart worker peaks at
# 2.03 GB (measured; see WORKER_BASELINE_MB in cloud/runner.py), so a full
# 96-way pool wants 195 GB. The first attempt ran on an n2-highcpu-96, which
# is 96 GB, and the kernel OOM-killed the campaign nine minutes in. highcpu
# gives 1 GB per vCPU and is the wrong family for this work; standard gives 4,
# which is 384 GB and leaves the pool a factor of two of headroom. The runner
# caps the pool from the same measurement, so a smaller machine now costs
# throughput rather than the run -- but it costs throughput exactly at n = 16,
# the size the campaign exists to measure.
#
# max-run-duration. Dropped, because the flag requires a termination action of
# DELETE and a MIG rejects that combination. It was there to bound the damage
# of a machine that had quietly wedged; that job now falls to the terminal
# marker and to `campaign.sh status`, which reports a run whose log has stopped
# advancing. A wedged machine therefore costs credit until someone looks,
# which is the one guarantee this change gives up.
# ---------------------------------------------------------------------------
