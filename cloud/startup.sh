#!/usr/bin/env bash
# Bring one campaign machine from a blank spot VM to a running campaign.
#
# GCE re-runs the startup script on EVERY boot, and with a MIG "every boot"
# includes every recreation after a preemption. So this script is written to
# be replayed: nothing here fails because it has already been done once. The
# state that must survive the machine lives in GCS, and the first thing a
# fresh instance does is pull it back -- results already computed, logs, and
# the per-restart cache that keeps a half-finished n = 16 restart from being
# thrown away.
#
# Order matters:
#     packages -> clone at the PINNED commits -> bootstrap (test_pq gate)
#     -> restore from GCS -> machine fingerprint -> sync loop -> runner
#     -> finisher (last sync, terminal marker, teardown)
#
# The fingerprint is written before any job starts because the trajectories
# are not bit-portable: a result is only comparable to the archive if the CPU
# flags and the package versions that produced it are on record.
#
# The runner is NOT the last step. It used to be, and a campaign that ran to
# completion then left a 96-vCPU spot machine idling behind a sync loop with
# nothing left to sync, recreated every 24 h by the MIG, until somebody
# noticed. Everything that has to happen when the runner returns lives in the
# finisher this script generates below.
#
# Usage: installed as the instance-template startup-script metadata by
# cloud/provision.sh. Never run by hand. Logs land in the serial console and
# in `journalctl -u google-startup-scripts`.
set -euo pipefail

WORKSPACE="${PQL_WORKSPACE:-/opt/pql}"
STATE_DIR="${PQL_STATE_DIR:-/var/lib/pql}"
ENV_FILE="${PQL_ENV_FILE:-/etc/pql/campaign.env}"

log() { printf '[pql-startup %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

meta() {
    curl -fsS -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/$1" 2>/dev/null \
        || printf '%s' "${2-}"
}
attr() { meta "instance/attributes/$1" "${2-}"; }

LANDSCAPE_URL="$(attr pql-landscape-url https://github.com/Ilyes-Jamoussi/peaked-circuit-landscape.git)"
LANDSCAPE_COMMIT="$(attr pql-landscape-commit)"
REPRO_URL="$(attr pql-repro-url https://github.com/Ilyes-Jamoussi/peaked-circuits-pennylane.git)"
REPRO_COMMIT="$(attr pql-repro-commit)"
BUCKET="$(attr pql-bucket)"
PREFIX="$(attr pql-prefix converged)"
BLOCK="$(attr pql-block converged_pilot)"
SIZES="$(attr pql-sizes)"
INSTANCES="$(attr pql-instances)"
WORKERS="$(attr pql-workers 0)"
JOBS="$(attr pql-jobs 1)"
SYNC_INTERVAL="$(attr pql-sync-interval 120)"
# Where runner.py puts its per-restart checkpoints. The runner owns this
# path (it passes --restart-cache to pq_experiment itself); it is declared
# here only so that the sync legs know what to carry.
RESTART_CACHE="$(attr pql-restart-cache results/restart_cache)"

REPO="$WORKSPACE/peaked-circuit-landscape"
REPRO="$WORKSPACE/peaked-circuits-pennylane"
VENV_PY="$WORKSPACE/campaign-env/bin/python"
GCS="$BUCKET/$PREFIX"
HOST="$(hostname)"

# One terminal marker per SHARD, not per host. A machine the MIG recreates
# comes back under a new name, so "has this work finished" cannot be keyed on
# whichever machine happened to be doing it; two machines sharding one block
# across --sizes must not answer for each other either. The slug is derived
# from the work, and cloud/campaign.sh recomputes it the same way.
shard_slug() {
    printf '%s' "$1${2:+-n$2}${3:+-i$3}" \
        | tr 'A-Z' 'a-z' | tr '_,' '--' | tr -cd 'a-z0-9.-'
}
SHARD="$(shard_slug "$BLOCK" "$SIZES" "$INSTANCES")"
MARKER="$GCS/status/$SHARD.txt"
FINISH="$WORKSPACE/pql-finish.sh"

mkdir -p "$WORKSPACE" "$STATE_DIR" "$(dirname "$ENV_FILE")"

# A boot that dies before the sync loop starts leaves no trace on the machine
# that GCP is about to delete. Push the reason to GCS instead.
report_failure() {
    local status=$?
    [ "$status" -eq 0 ] && return 0
    log "FAILED with status $status"
    if [ -n "$BUCKET" ] && command -v gcloud >/dev/null 2>&1; then
        {
            printf 'host %s\ndate %s\nstatus %s\n' \
                "$HOST" "$(date -u +%FT%TZ)" "$status"
            printf 'landscape %s\nreproduction %s\nblock %s\n' \
                "$LANDSCAPE_COMMIT" "$REPRO_COMMIT" "$BLOCK"
            journalctl -u google-startup-scripts --no-pager -n 200 2>/dev/null || true
        } | gcloud storage cp - "$GCS/logs/$HOST/BOOT_FAILED_$(date -u +%Y%m%dT%H%M%SZ).txt" \
            >/dev/null 2>&1 || true
    fi
    return "$status"
}
trap report_failure EXIT

[ -n "$LANDSCAPE_COMMIT" ] || { log "no pql-landscape-commit in metadata"; exit 1; }
[ -n "$REPRO_COMMIT" ] || { log "no pql-repro-commit in metadata"; exit 1; }
[ -n "$BUCKET" ] || { log "no pql-bucket in metadata"; exit 1; }

case "$WORKERS" in ''|0|*[!0-9]*) WORKERS="$(nproc)" ;; esac
case "$JOBS" in ''|0|*[!0-9]*) JOBS=1 ;; esac

log "block=$BLOCK sizes='${SIZES:-all}' instances='${INSTANCES:-all}' workers=$WORKERS jobs=$JOBS"
log "pinned landscape=$LANDSCAPE_COMMIT reproduction=$REPRO_COMMIT"

# ---------------------------------------------------------------- packages
export DEBIAN_FRONTEND=noninteractive
for attempt in 1 2 3; do
    apt-get update -y >/dev/null 2>&1 && break
    log "apt-get update failed (attempt $attempt), retrying"
    sleep $((attempt * 10))
done
apt-get install -y --no-install-recommends \
    git curl rsync python3-venv python3-dev build-essential >/dev/null 2>&1 \
    || log "apt-get install reported an error; continuing if the tools are present"

if ! command -v gcloud >/dev/null 2>&1; then
    log "installing google-cloud-cli"
    apt-get install -y google-cloud-cli >/dev/null 2>&1 || {
        log "google-cloud-cli not in the image repositories"; exit 1;
    }
fi
command -v git >/dev/null 2>&1 || { log "git missing after install"; exit 1; }

# ------------------------------------------------------------------ clones
clone_pinned() {
    local url="$1" dir="$2" commit="$3" attempt
    if [ ! -d "$dir/.git" ]; then
        for attempt in 1 2 3; do
            git clone --quiet "$url" "$dir" && break
            log "clone of $url failed (attempt $attempt)"
            rm -rf "$dir"
            sleep $((attempt * 10))
        done
    fi
    [ -d "$dir/.git" ] || { log "could not clone $url"; return 1; }
    git -C "$dir" fetch --quiet --all --tags || true
    if ! git -C "$dir" rev-parse --verify --quiet "${commit}^{commit}" >/dev/null; then
        log "commit $commit is absent from $url: refusing to run a different version"
        return 1
    fi
    git -C "$dir" checkout --quiet --detach "$commit"
    log "$(basename "$dir") at $(git -C "$dir" rev-parse --short HEAD)"
}

clone_pinned "$LANDSCAPE_URL" "$REPO" "$LANDSCAPE_COMMIT"
clone_pinned "$REPRO_URL" "$REPRO" "$REPRO_COMMIT"

# ------------------------------------------------- environment and the gate
# bootstrap.sh builds campaign-env, runs the twelve self-tests of test_pq.py
# (the seven original sanity checks, the four that guard the frozen protocol
# against the converged instrumentation, and the one that holds the pool's
# memory model to the measurement it was fitted to) and the mini end-to-end
# job. It is cheap next to the campaign and it is the only check that this
# particular machine computes what the archive says it should, so it runs on
# every boot rather than once.
log "running bootstrap.sh (venv, test_pq.py, mini dry run)"
bash "$REPO/cloud/bootstrap.sh"
[ -x "$VENV_PY" ] || { log "bootstrap did not produce $VENV_PY"; exit 1; }

# ----------------------------------------------------------- restore state
# No --delete-unmatched-destination-objects anywhere below: this is a
# restore, not a mirror, and the local tree also holds the mini artifacts
# bootstrap just wrote.
restore() {
    local src="$1" dst="$2" attempt
    if ! gcloud storage ls "$src" >/dev/null 2>&1; then
        log "nothing to restore from $src"
        return 0
    fi
    mkdir -p "$dst"
    for attempt in 1 2 3; do
        if gcloud storage rsync --recursive "$src" "$dst" >/dev/null 2>&1; then
            log "restored $src -> $dst"
            return 0
        fi
        log "restore from $src failed (attempt $attempt)"
        sleep $((attempt * 10))
    done
    # Fatal on purpose: continuing would recompute work the bucket already
    # holds, at n = 16 prices, and would do it silently.
    log "could not restore $src after three attempts"
    return 1
}

restore "$GCS/results" "$REPO/results"
restore "$GCS/analysis" "$REPO/analysis"
# The restart cache sits under results/ by default, so the first leg already
# carries it. The third leg only matters if it is ever moved out.
case "$RESTART_CACHE" in
    results/*) ;;
    *) restore "$GCS/restart-cache" "$REPO/$RESTART_CACHE" ;;
esac

# ------------------------------------------------------ machine fingerprint
# The remedy for trajectories that are not bit-portable: whatever this
# machine produces can be attributed to this exact CPU and package set.
FINGERPRINT="$REPO/results/$BLOCK/ENV_${HOST}.txt"
mkdir -p "$(dirname "$FINGERPRINT")"
{
    printf 'host            %s\n' "$HOST"
    printf 'written         %s\n' "$(date -u +%FT%TZ)"
    printf 'zone            %s\n' "$(basename "$(meta instance/zone)")"
    printf 'machine-type    %s\n' "$(basename "$(meta instance/machine-type)")"
    printf 'instance-id     %s\n' "$(meta instance/id)"
    printf 'block           %s\n' "$BLOCK"
    printf 'sizes           %s\n' "${SIZES:-all}"
    printf 'instances       %s\n' "${INSTANCES:-all}"
    printf 'workers         %s\n' "$WORKERS"
    printf 'jobs            %s\n' "$JOBS"
    printf 'landscape-sha   %s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'reproduction-sha %s\n' "$(git -C "$REPRO" rev-parse HEAD)"
    printf '\n--- uname -a ---\n'
    uname -a
    printf '\n--- lscpu ---\n'
    lscpu 2>/dev/null || true
    printf '\n--- /proc/cpuinfo flags ---\n'
    grep -m1 '^flags' /proc/cpuinfo || true
    printf '\n--- /proc/cpuinfo model ---\n'
    grep -m1 '^model name' /proc/cpuinfo || true
    printf '\n--- pip freeze ---\n'
    "$WORKSPACE/campaign-env/bin/pip" freeze 2>/dev/null || true
    printf '\n--- python ---\n'
    "$VENV_PY" -VV
} > "$FINGERPRINT"
log "fingerprint written to $FINGERPRINT"

# ---------------------------------------------------- shared configuration
# sync.sh runs from the clone and shutdown.sh runs from metadata; neither can
# read the other's variables, so both read this file.
cat > "$ENV_FILE" <<EOF
PQL_WORKSPACE=$WORKSPACE
PQL_REPO=$REPO
PQL_BUCKET=$BUCKET
PQL_PREFIX=$PREFIX
PQL_GCS=$GCS
PQL_BLOCK=$BLOCK
PQL_HOST=$HOST
PQL_RESTART_CACHE=$RESTART_CACHE
PQL_SYNC_INTERVAL=$SYNC_INTERVAL
PQL_STATE_DIR=$STATE_DIR
PQL_SHARD=$SHARD
PQL_MARKER=$MARKER
EOF

# ------------------------------------------------------------- the finisher
# Generated here rather than shipped in cloud/ because only startup.sh knows
# it is wanted, and because a machine must never run a finisher newer than
# the pinned commit that started its campaign.
cat > "$FINISH" <<'PQL_FINISH_EOF'
#!/usr/bin/env bash
# Close out one campaign run: last sync, terminal marker, teardown.
#
# Written at boot by cloud/startup.sh and invoked once, by the pql-campaign
# unit, with the runner's exit status as $1. Nothing else calls it.
#
# It exists because "the runner returned" was previously not an event. The
# runner exited, systemd collected the unit, and a 96-vCPU spot machine went
# on billing behind a sync loop with nothing left to sync -- recreated every
# 24 h by --max-run-duration -- until a human ran `campaign.sh down`. On a
# ~284 $ credit that is the difference between a finished campaign and a
# finished budget.
#
# The order is the whole design and it is not negotiable:
#
#     1. stop the periodic sync loop     nothing may race the final pass
#     2. one full synchronous pass       the DATA
#     3. write the terminal marker       the CLAIM about the data
#     4. tear down, on success only      the MACHINE
#
# A marker written before the data would announce a campaign whose archives
# are not in the bucket, and `campaign.sh status` would report success over an
# incomplete mirror. The teardown is last because it deletes this instance:
# anything sequenced after it may simply never run.
#
# Two more rules follow from the same reasoning:
#
#   * Teardown requires state DONE *and* a final sync that passed. A campaign
#     whose last pass failed may hold results only on this disk, and this disk
#     is deleted with the instance.
#   * Success and failure end differently. A run that failed stops syncing,
#     writes FAILED, and stays up: its logs are what the next attempt needs,
#     and destroying them to save a few dollars is the wrong trade. A machine
#     the MIG has since rebuilt is a different case -- the failed run's logs
#     are already in the bucket -- and that boot does tear down.
#
# Deliberately NOT `set -e`. Every step is checked by hand precisely because
# the script must reach the teardown even when the step before it failed; an
# abort halfway through is the exact failure this file exists to prevent.
# `set -u` still holds: an unset variable here would silently address the
# wrong bucket path.
#
# Overrides, all read from /etc/pql/campaign.env if the metadata server
# cannot answer: PQL_MIG, PQL_ZONE, PQL_PROJECT.
set -uo pipefail

EXIT_CODE="${1:-1}"
case "$EXIT_CODE" in
    ''|*[!0-9]*) EXIT_CODE=1 ;;
esac

ENV_FILE="${PQL_ENV_FILE:-/etc/pql/campaign.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE"
fi

REPO="${PQL_REPO:-/opt/pql/peaked-circuit-landscape}"
GCS="${PQL_GCS:-${PQL_BUCKET:-}/${PQL_PREFIX:-converged}}"
HOST="${PQL_HOST:-$(hostname)}"
BLOCK="${PQL_BLOCK:-unknown}"
SHARD="${PQL_SHARD:-$BLOCK}"
MARKER="${PQL_MARKER:-$GCS/status/$SHARD.txt}"
ALERT="$GCS/status/ACTION_REQUIRED_$SHARD.txt"
# Set by startup.sh when this boot found a terminal marker already in the
# bucket, i.e. this machine is a replacement for one that had finished.
REBOOT="${PQL_MARKER_REBOOT:-0}"

log() { printf '[pql-finish %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Wide enough to survive `tail -20` and a skim of campaign.out. If a human
# has to act, this is what they are meant to trip over.
loud() {
    printf '\n'
    printf '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
    while [ $# -gt 0 ]; do printf '!! %s\n' "$1"; shift; done
    printf '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
    printf '\n'
}

meta() {
    curl -fsS -H 'Metadata-Flavor: Google' \
        "http://metadata.google.internal/computeMetadata/v1/$1" 2>/dev/null \
        || printf '%s' "${2-}"
}

# Which group owns this instance. `created-by` is set by the MIG itself and
# reads projects/<n>/zones/<zone>/instanceGroupManagers/<name>; deriving the
# name from PQL_BLOCK instead would guess at provision.sh's naming and get it
# wrong for any non-default PQL_NAME_PREFIX.
MIG_NAME="${PQL_MIG:-}"
if [ -z "$MIG_NAME" ]; then
    CREATED_BY="$(meta instance/attributes/created-by)"
    case "$CREATED_BY" in
        */instanceGroupManagers/*) MIG_NAME="${CREATED_BY##*/}" ;;
        *) MIG_NAME="" ;;
    esac
fi
ZONE="${PQL_ZONE:-}"
if [ -z "$ZONE" ]; then
    ZONE="$(meta instance/zone)"
    ZONE="${ZONE##*/}"
fi
PROJECT="${PQL_PROJECT:-}"
if [ -z "$PROJECT" ]; then
    PROJECT="$(meta project/project-id)"
fi

if [ "$EXIT_CODE" -eq 0 ]; then STATE=DONE; else STATE=FAILED; fi
FINISHED_AT="$(date -u +%FT%TZ)"
SYNC_STATE=unknown

HAVE_BUCKET=1
case "$GCS" in
    gs://*) ;;
    *) HAVE_BUCKET=0 ;;
esac

log "runner exited with status $EXIT_CODE -> $STATE (shard $SHARD)"
if [ "$HAVE_BUCKET" -ne 1 ]; then
    loud "no bucket configured in $ENV_FILE: no final sync, no marker." \
         "This machine is still up and still billing. Take it down by hand."
    exit "$EXIT_CODE"
fi

push_narration() {
    [ -f "$REPO/campaign.out" ] || return 0
    gcloud storage cp "$REPO/campaign.out" "$GCS/logs/$HOST/campaign.out" \
        >/dev/null 2>&1 || log "could not push campaign.out"
}

# The marker is the answer `campaign.sh status` reads. Plain key/value, one
# object per shard, rewritten in place: the last writer is the current truth.
write_marker() {
    local teardown="$1" warning="${2:-}"
    {
        printf 'state      %s\n' "$STATE"
        printf 'exit       %s\n' "$EXIT_CODE"
        printf 'block      %s\n' "$BLOCK"
        printf 'shard      %s\n' "$SHARD"
        printf 'host       %s\n' "$HOST"
        printf 'finished   %s\n' "$FINISHED_AT"
        printf 'written    %s\n' "$(date -u +%FT%TZ)"
        printf 'sync       %s\n' "$SYNC_STATE"
        printf 'teardown   %s\n' "$teardown"
        printf 'mig        %s\n' "${MIG_NAME:-none}"
        printf 'zone       %s\n' "${ZONE:-unknown}"
        printf 'project    %s\n' "${PROJECT:-unknown}"
        if [ -n "$warning" ]; then printf 'warning    %s\n' "$warning"; fi
    } | gcloud storage cp - "$MARKER" >/dev/null 2>&1 \
        || log "could not write the marker $MARKER"
}

# A second object, under a name that shouts in a plain `gcloud storage ls`.
# Written only when a human has to do something about a running machine.
write_alert() {
    local reason="$1"
    {
        printf 'ACTION REQUIRED -- this campaign machine is still billing\n\n'
        printf 'shard    %s\n' "$SHARD"
        printf 'block    %s\n' "$BLOCK"
        printf 'host     %s\n' "$HOST"
        printf 'state    %s\n' "$STATE"
        printf 'when     %s\n' "$(date -u +%FT%TZ)"
        printf 'reason   %s\n\n' "$reason"
        printf 'The campaign has stopped and the sync loop is down, so nothing\n'
        printf 'more will reach this bucket. The instance did not take itself\n'
        printf 'out. Collect what is here and tear the group down by hand:\n\n'
        printf '    cloud/campaign.sh fetch\n'
        printf '    cloud/campaign.sh down\n'
    } | gcloud storage cp - "$ALERT" >/dev/null 2>&1 \
        || log "could not write the alert $ALERT"
}

# ------------------------------------------------- 1. stop the periodic loop
# Restart=always does not fight an explicit `systemctl stop`. Stopping first
# means the final pass below has the bucket to itself; leaving the loop up
# would also keep re-uploading a tree that has stopped changing, forever.
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet pql-sync; then
    if systemctl stop pql-sync >/dev/null 2>&1; then
        log "periodic sync loop stopped"
    else
        log "could not stop pql-sync; the final pass may race it"
    fi
else
    log "no running pql-sync unit to stop"
fi

# ------------------------------------------------------ 2. the final pass
# A full pass, not the newer-than-stamp scan shutdown.sh does: there is no
# 30-second clock here, and this is the copy the marker is about to vouch for.
if [ -f "$REPO/cloud/sync.sh" ]; then
    for attempt in 1 2 3; do
        if PQL_ENV_FILE="$ENV_FILE" /bin/bash "$REPO/cloud/sync.sh" --once; then
            SYNC_STATE=ok
            break
        fi
        SYNC_STATE=FAILED
        log "final sync pass failed (attempt $attempt)"
        sleep $((attempt * 10))
    done
else
    SYNC_STATE=missing
    log "no $REPO/cloud/sync.sh: this boot cannot make a final pass"
fi
log "final sync: $SYNC_STATE"

# ---------------------------------------------------- 3. decide, then record
mig_reachable() {
    local attempt
    for attempt in 1 2; do
        if gcloud compute instance-groups managed describe "$MIG_NAME" \
                --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
            return 0
        fi
        log "cannot describe $MIG_NAME (attempt $attempt)"
        sleep 5
    done
    return 1
}

TEARDOWN=pending
WARNING=""
if [ "$STATE" != DONE ] && [ "$REBOOT" != 1 ]; then
    TEARDOWN=kept-for-diagnosis
    WARNING="the campaign FAILED (exit $EXIT_CODE); this machine is kept up on purpose so its logs can be read, and it is still billing"
elif [ "$SYNC_STATE" != ok ]; then
    TEARDOWN=kept-sync-incomplete
    WARNING="the final sync ended '$SYNC_STATE'; refusing to destroy a disk that may hold the only copy of these results"
elif [ -z "$MIG_NAME" ]; then
    TEARDOWN=poweroff
elif mig_reachable; then
    TEARDOWN=resize
else
    TEARDOWN=denied
    WARNING="this instance cannot read its own managed group $MIG_NAME, so it cannot resize it to 0 and the group will keep recreating this machine; see cloud/IAM.md, this is a missing compute.instanceAdmin role or a scope narrower than cloud-platform"
fi

if [ "$STATE" != DONE ] && [ "$REBOOT" = 1 ]; then
    log "this machine is a replacement for a run that already ended $STATE;"
    log "its logs are in the bucket, so the teardown applies here"
fi
log "teardown plan: $TEARDOWN"
if [ -n "$WARNING" ]; then
    loud "$WARNING" \
         "" \
         "shard $SHARD, host $HOST, state $STATE, exit $EXIT_CODE" \
         "run:  cloud/campaign.sh fetch  &&  cloud/campaign.sh down"
fi
push_narration

# --------------------------------------------------------------- 4. act
case "$TEARDOWN" in
    resize)
        # Recorded before the call, because the call deletes this instance and
        # the rewrite below may never run. `resize-requested` in the marker
        # with targetSize 0 on the group means it landed.
        write_marker resize-requested ""
        log "resizing $MIG_NAME to 0 in $ZONE: this instance is next"
        if gcloud compute instance-groups managed resize "$MIG_NAME" \
                --project="$PROJECT" --zone="$ZONE" --size=0 --quiet; then
            log "group $MIG_NAME resized to 0; the campaign is over"
            push_narration
            write_marker resized ""
        else
            WARNING="the resize of $MIG_NAME to 0 was refused; this machine is still up and still billing"
            loud "$WARNING" \
                 "" \
                 "run:  cloud/campaign.sh fetch  &&  cloud/campaign.sh down"
            push_narration
            write_marker resize-FAILED "$WARNING"
            write_alert "$WARNING"
        fi
        ;;
    poweroff)
        # No managed group owns this instance, so nothing will recreate it and
        # a plain power off is the whole teardown.
        write_marker poweroff ""
        push_narration
        log "no managed group in the metadata: powering off"
        poweroff || shutdown -h now || log "could not power this instance off"
        ;;
    *)
        write_marker "$TEARDOWN" "$WARNING"
        if [ -n "$WARNING" ]; then write_alert "$WARNING"; fi
        push_narration
        ;;
esac

exit "$EXIT_CODE"
PQL_FINISH_EOF
chmod +x "$FINISH"

# The finisher is generated text, so `bash -n cloud/startup.sh` says nothing
# about it. Parse it here: a finisher that does not run is a campaign with no
# end, which is the whole point of the file.
bash -n "$FINISH" || { log "the generated finisher does not parse"; exit 1; }
log "finisher written to $FINISH (marker $MARKER)"

# ------------------------------------------------------------- the watchdog
# The finisher is invoked by the campaign unit's own wrapper shell, which
# means it only runs when that shell survives long enough to call it. It did
# not, once: a kernel OOM kill inside the unit's cgroup took the wrapper with
# it, and the campaign simply stopped existing -- no marker, no final sync, no
# teardown, and a 96-vCPU spot machine left billing behind a sync loop.
# OOMPolicy=continue on the unit below closes that particular hole; this
# closes the class. Anything that can end the campaign unit without running
# the finisher -- an OOM, a SIGKILL, an operator stopping the unit -- is
# caught here.
#
# The predicate is exact rather than heuristic, and that is the whole design:
# the finisher's FIRST action is to stop pql-sync. So
#
#     campaign unit gone  AND  sync loop still up
#
# holds if and only if the campaign ended and the finisher did not run. While
# the finisher is working the campaign unit is still active (the finisher runs
# inside it), so there is no window in which this fires by mistake.
WATCHDOG="$WORKSPACE/pql-watchdog.sh"
cat > "$WATCHDOG" <<'PQL_WATCHDOG_EOF'
#!/usr/bin/env bash
set -uo pipefail

ENV_FILE="${PQL_ENV_FILE:-/etc/pql/campaign.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE"
fi
WORKSPACE="${PQL_WORKSPACE:-/opt/pql}"
REPO="${PQL_REPO:-$WORKSPACE/peaked-circuit-landscape}"
FINISH="$WORKSPACE/pql-finish.sh"
INTERVAL="${PQL_WATCHDOG_INTERVAL:-60}"

log() { printf '[pql-watchdog %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Do not judge before there is anything to judge: this unit starts before the
# campaign does, and "never started" must not read as "ended without a
# finisher".
seen=0
while :; do
    sleep "$INTERVAL"
    if systemctl is-active --quiet pql-campaign; then
        seen=1
        continue
    fi
    [ "$seen" -eq 1 ] || continue
    if ! systemctl is-active --quiet pql-sync; then
        log "campaign ended and the sync loop is down: the finisher ran"
        exit 0
    fi
    # --collect drops the unit once it is inactive, so the exit status is
    # often already gone. It does not matter: reaching this line means the
    # campaign died without closing itself out, which is a failure whatever
    # the number was.
    status="$(systemctl show -p ExecMainStatus --value pql-campaign 2>/dev/null)"
    case "$status" in ''|0|*[!0-9]*) status=1 ;; esac
    log "campaign unit is gone and the sync loop is still up:"
    log "the finisher did not run. Invoking it with status $status."
    {
        printf '\n=== watchdog %s ===\n' "$(date -u +%FT%TZ)"
        printf '=== the campaign unit ended without invoking the finisher\n'
        printf '=== (OOM kill, SIGKILL or an explicit stop). Closing out.\n'
    } >> "$REPO/campaign.out"
    PQL_ENV_FILE="$ENV_FILE" /bin/bash "$FINISH" "$status" \
        >> "$REPO/campaign.out" 2>&1
    exit 0
done
PQL_WATCHDOG_EOF
chmod +x "$WATCHDOG"
bash -n "$WATCHDOG" || { log "the generated watchdog does not parse"; exit 1; }

# --------------------------------------------------------------- processes
# systemd-run, not nohup: the startup script runs inside a oneshot unit whose
# cgroup is torn down when it exits, which kills any plain background child.
#
# systemd-run succeeds as soon as the unit is queued, so a missing script
# would be reported as a running sync loop that mirrors nothing.
[ -f "$REPO/cloud/sync.sh" ] || {
    log "no cloud/sync.sh at $LANDSCAPE_COMMIT: nothing would reach $GCS"; exit 1;
}

# ------------------------------------------------------ already finished?
# This script is replayed on every boot, and with --max-run-duration the MIG
# rebuilds this machine every 24 h whether or not there is work left. A shard
# that has already written its terminal marker must not be started again: it
# would re-run a campaign whose jobs all skip, rewrite the marker, and keep a
# 96-vCPU spot machine on the meter in between. Hand the boot to the finisher
# instead, which re-attempts the teardown that evidently did not take.
#
# To force a re-run, delete the marker:  gcloud storage rm <marker>
MARKER_BODY=""
if ! systemctl is-active --quiet pql-campaign; then
    MARKER_BODY="$(gcloud storage cat "$MARKER" 2>/dev/null || true)"
fi
MARKER_STATE="$(printf '%s\n' "$MARKER_BODY" | awk '$1 == "state" { print $2; exit }')"
MARKER_EXIT="$(printf '%s\n' "$MARKER_BODY" | awk '$1 == "exit" { print $2; exit }')"
case "$MARKER_STATE" in
    DONE|FAILED)
        log "shard $SHARD is already terminal ($MARKER_STATE) in $MARKER"
        log "not relaunching the campaign; re-running the finisher instead"
        {
            printf '\n=== boot %s on %s ===\n' "$(date -u +%FT%TZ)" "$HOST"
            printf '=== shard %s already ended %s; this machine should not\n' \
                "$SHARD" "$MARKER_STATE"
            printf '=== exist. Closing it out again rather than re-running.\n'
            printf '=== To force a re-run: gcloud storage rm %s\n' "$MARKER"
        } >> "$REPO/campaign.out"
        PQL_ENV_FILE="$ENV_FILE" PQL_MARKER_REBOOT=1 \
            /bin/bash "$FINISH" "${MARKER_EXIT:-0}" >> "$REPO/campaign.out" 2>&1 \
            || log "finisher returned non-zero (the recorded campaign had failed)"
        log "startup complete (shard already terminal)"
        exit 0
        ;;
esac

if systemctl is-active --quiet pql-sync; then
    log "sync loop already running"
else
    systemd-run --unit=pql-sync --collect \
        --property=Restart=always --property=RestartSec=30 \
        --working-directory="$REPO" \
        /bin/bash "$REPO/cloud/sync.sh" >/dev/null
    log "sync loop started (every ${SYNC_INTERVAL}s)"
fi

if systemctl is-active --quiet pql-watchdog; then
    log "watchdog already running"
else
    systemd-run --unit=pql-watchdog --collect \
        --property=Restart=on-failure --property=RestartSec=30 \
        --setenv=PQL_ENV_FILE="$ENV_FILE" \
        --working-directory="$REPO" \
        /bin/bash "$WATCHDOG" >/dev/null
    log "watchdog started"
fi

if systemctl is-active --quiet pql-campaign; then
    log "campaign already running; leaving it alone"
else
    RUNNER_ARGS=(cloud/runner.py --only "$BLOCK" --workers "$WORKERS")
    if [ -n "$SIZES" ]; then RUNNER_ARGS+=(--sizes "$SIZES"); fi
    if [ -n "$INSTANCES" ]; then RUNNER_ARGS+=(--instances "$INSTANCES"); fi
    # --jobs is newer than some pinned commits, and a pin is exactly what this
    # machine runs: ask the runner what it accepts rather than assume.
    if [ "$JOBS" -gt 1 ]; then
        if "$VENV_PY" "$REPO/cloud/runner.py" --help 2>&1 | grep -q -- '--jobs'; then
            RUNNER_ARGS+=(--jobs "$JOBS")
        else
            log "runner.py at this pin has no --jobs; running sequentially"
        fi
    fi
    printf -v RUNNER_CMD '%q ' "$VENV_PY" "${RUNNER_ARGS[@]}"
    log "launching: $RUNNER_CMD"
    {
        printf '\n=== boot %s on %s ===\n' "$(date -u +%FT%TZ)" "$HOST"
        printf '=== landscape %s reproduction %s ===\n' \
            "$(git -C "$REPO" rev-parse --short HEAD)" \
            "$(git -C "$REPRO" rev-parse --short HEAD)"
        printf '=== %s\n' "$RUNNER_CMD"
    } >> "$REPO/campaign.out"
    # No `exec` any more: the runner's exit IS the end of the campaign, and
    # something has to still be alive to notice it. The wrapper hands the
    # runner's status straight to the finisher, so the unit stays active until
    # the last sync, the marker and the teardown are done. "$?" is literal
    # here -- it is expanded by the shell systemd-run starts, not by this one.
    printf -v CAMPAIGN_CMD '%s>> %q 2>&1; /bin/bash %q "$?" >> %q 2>&1' \
        "$RUNNER_CMD" "$REPO/campaign.out" "$FINISH" "$REPO/campaign.out"
    # OOMPolicy=continue is load-bearing. systemd's default for a service is
    # `stop`: when the kernel OOM-kills ONE process in the unit's cgroup,
    # systemd tears the whole cgroup down -- including the wrapper shell that
    # is supposed to call the finisher. That is exactly how the first n = 16
    # attempt ended: 96 workers asked for 195 GB on a 96 GB machine, a worker
    # was killed, the unit was collected with it, and no marker, no final sync
    # and no teardown ever happened. The machine sat there billing behind a
    # sync loop with an empty bucket. With `continue`, the kill is reported to
    # the runner as a broken pool, the runner exits non-zero, and the finisher
    # runs and records FAILED -- which is what a dead campaign should look
    # like from the bucket.
    systemd-run --unit=pql-campaign --collect \
        --property=OOMPolicy=continue \
        --working-directory="$REPO" \
        --setenv=PQL_ENV_FILE="$ENV_FILE" \
        --setenv=PEAKED_REPRO_PATH="$REPRO" \
        --setenv=OMP_NUM_THREADS=1 \
        --setenv=OPENBLAS_NUM_THREADS=1 \
        --setenv=MKL_NUM_THREADS=1 \
        /bin/bash -c "$CAMPAIGN_CMD" >/dev/null
    log "campaign started under systemd unit pql-campaign"
    log "on exit it will sync, write $MARKER, and take the machine down"
fi

log "startup complete"
