#!/usr/bin/env bash
# The single door to the cloud campaign: up, status, logs, fetch, down.
#
# One entry point exists so that the sequence that matters cannot be got
# wrong. The last campaign ended with a manual teardown that deleted the only
# copy of its logs, so `down` here refuses to delete anything until a fetch
# has actually brought the bucket home and the bucket holds nothing newer.
#
# Usage:
#     cloud/campaign.sh up                  # template + managed group, size 1
#     cloud/campaign.sh status              # group, instance, job progress
#     cloud/campaign.sh logs [-f] [--host H]
#     cloud/campaign.sh fetch               # GCS -> laptop, then pq_validate
#     cloud/campaign.sh down [--force]      # group then template, after fetch
#
# Each command is a few seconds except fetch, which moves the archives.
set -euo pipefail

PQL_PROJECT="${PQL_PROJECT:-project-b2dccd70-9346-4730-9ef}"
PQL_ZONE="${PQL_ZONE:-northamerica-northeast1-b}"
PQL_BUCKET="${PQL_BUCKET:-gs://${PQL_PROJECT}-pql-campaign}"
PQL_PREFIX="${PQL_PREFIX:-converged}"
PQL_BLOCK="${PQL_BLOCK:-converged_pilot}"
PQL_SIZES="${PQL_SIZES:-}"
PQL_INSTANCES="${PQL_INSTANCES:-}"
PQL_WORKERS="${PQL_WORKERS:-96}"
PQL_JOBS="${PQL_JOBS:-1}"
PQL_NAME_PREFIX="${PQL_NAME_PREFIX:-pql}"
PQL_LOCAL_LOGS="${PQL_LOCAL_LOGS:-$HOME/pql-campaign-logs}"
PQL_LOCAL_STATE="${PQL_LOCAL_STATE:-$HOME/.pql-campaign}"
PQL_PYTHON="${PQL_PYTHON:-}"           # empty: look for one that has numpy
PQL_FOLLOW_INTERVAL="${PQL_FOLLOW_INTERVAL:-15}"

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$CLOUD_DIR")"
GCS="$PQL_BUCKET/$PQL_PREFIX"
BLOCK_SLUG="$(printf '%s' "$PQL_BLOCK" | tr 'A-Z' 'a-z' | tr '_' '-' | tr -cd 'a-z0-9-')"
MIG_NAME="${PQL_NAME_PREFIX}-${BLOCK_SLUG}"
FETCH_STAMP="$PQL_LOCAL_STATE/${MIG_NAME}.fetch"

# Must stay byte-identical to shard_slug() in cloud/startup.sh: that is the
# machine that writes the marker, this is the machine that reads it, and a
# slug that disagrees turns "finished" into "still running" on this side.
shard_slug() {
    printf '%s' "$1${2:+-n$2}${3:+-i$3}" \
        | tr 'A-Z' 'a-z' | tr '_,' '--' | tr -cd 'a-z0-9.-'
}
SHARD="$(shard_slug "$PQL_BLOCK" "$PQL_SIZES" "$PQL_INSTANCES")"
MARKER="$GCS/status/$SHARD.txt"

die() { printf 'campaign: %s\n' "$*" >&2; exit 1; }
note() { printf 'campaign: %s\n' "$*"; }
rule() { printf -- '\n--- %s ---------------------------------------\n' "$1"; }

usage() {
    cat <<EOF
cloud/campaign.sh COMMAND [options]

Commands
  up                  create the instance template and the managed instance
                      group (size 1, spot, auto-recreated on preemption)
  status              the verdict first -- running, finished or failed, read
                      from the shard's terminal marker in the bucket -- then
                      group state, instance, MIG errors and the tail of
                      campaign.out
  logs [-f] [--host H]
                      print campaign.out from GCS; -f follows every ${PQL_FOLLOW_INTERVAL}s
  fetch               mirror results, analysis logs and run logs from GCS to
                      this machine, then run pq_validate.py --allow-missing
  down [--force]      delete the group then the template. Refuses unless a
                      fetch has succeeded and the bucket holds nothing newer.
                      --force accepts a fetch whose gate failed, and a
                      still-running campaign whose newest objects are not home
                      yet (they stay in the bucket; fetch again afterwards).
                      Nothing forces past a fetch that never happened.

A campaign that finishes cleanly now takes its own machine down: the VM makes
a last full sync, writes the marker, and resizes the group to 0. \`down\` is
still the manual net -- it is what deletes the (free, empty) group and its
template, and it is the only way out when the VM lacks the compute rights to
resize its own group, or when the run failed and was kept up for diagnosis.
Both of those cases shout in \`status\`.

  marker            $MARKER

Configuration (environment, with the defaults in force here)
  PQL_PROJECT       $PQL_PROJECT
  PQL_ZONE          $PQL_ZONE
  PQL_BUCKET        $PQL_BUCKET
  PQL_PREFIX        $PQL_PREFIX
  PQL_BLOCK         $PQL_BLOCK        (runner.py --only)
  PQL_SIZES         ${PQL_SIZES:-<all>}   (runner.py --sizes, shards a block)
  PQL_INSTANCES     ${PQL_INSTANCES:-<all>}   (runner.py --instances)
  PQL_WORKERS       $PQL_WORKERS
  PQL_JOBS          $PQL_JOBS         (runner.py --jobs, concurrent runs)
  PQL_LOCAL_LOGS    $PQL_LOCAL_LOGS
  PQL_LOCAL_STATE   $PQL_LOCAL_STATE
  PQL_PYTHON        ${PQL_PYTHON:-<auto: $(pick_python)>}

No credentials are read or stored here: everything authenticates through the
active gcloud account on this machine and, on the VM, through the instance
service account with the storage-rw scope.
EOF
}

require_gcloud() {
    command -v gcloud >/dev/null 2>&1 || die "gcloud not on PATH"
}

# The integrity gate needs numpy, and the interpreter that has it is a venv
# beside the repository, not the system python3. Guessing wrong would make
# every fetch report a failed gate for want of an import.
pick_python() {
    if [ -n "$PQL_PYTHON" ]; then printf '%s' "$PQL_PYTHON"; return; fi
    local candidate
    for candidate in \
        "$REPO_DIR/.venv/bin/python" \
        "$REPO_DIR/../campaign-env/bin/python" \
        "$REPO_DIR/../quantum-env/bin/python" \
        "$REPO_DIR/../../campaign-env/bin/python" \
        "$REPO_DIR/../../quantum-env/bin/python" \
        "$(command -v python3 || true)"
    do
        if [ -x "$candidate" ] && "$candidate" -c 'import numpy' >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return
        fi
    done
    printf '%s' "${PQL_PYTHON:-python3}"
}

mig_exists() {
    gcloud compute instance-groups managed describe "$MIG_NAME" \
        --project="$PQL_PROJECT" --zone="$PQL_ZONE" >/dev/null 2>&1
}

# The host whose narration is freshest; falls back to the last one listed.
default_host() {
    local newest
    newest="$(gcloud storage ls --long "$GCS/logs/**/campaign.out" 2>/dev/null \
        | awk 'NF >= 3 && $NF ~ /campaign\.out$/ { print $(NF-1), $NF }' \
        | sort | tail -1 \
        | sed -n 's#.*/logs/\([^/]*\)/campaign.out$#\1#p')"
    if [ -n "$newest" ]; then
        printf '%s' "$newest"
        return
    fi
    gcloud storage ls "$GCS/logs/" 2>/dev/null \
        | sed -n 's#.*/logs/\([^/]*\)/$#\1#p' | tail -1
}

cmd_up() {
    require_gcloud
    export PQL_PROJECT PQL_ZONE PQL_BUCKET PQL_PREFIX PQL_BLOCK \
        PQL_SIZES PQL_INSTANCES PQL_WORKERS PQL_JOBS PQL_NAME_PREFIX
    exec bash "$CLOUD_DIR/provision.sh"
}

# The verdict, not the vibe.
#
# The tail of campaign.out cannot answer "is this campaign over": a finished
# run, a wedged run and a run whose machine was preempted thirty seconds ago
# all look the same from there -- lines stop. The marker is written by the VM
# itself, once, after its last full sync pass, and is the only statement in
# the system that distinguishes them.
#
# Absence of a marker is meaningful too, and it means RUNNING: a preempted
# machine writes none, and its replacement resumes the shard.
report_state() {
    local body state teardown warning line
    body="$(gcloud storage cat "$MARKER" 2>/dev/null || true)"

    if [ -z "$body" ]; then
        printf 'STATE: RUNNING -- no terminal marker for this shard yet\n'
        printf '  shard   %s\n' "$SHARD"
        printf '  marker  %s (absent)\n' "$MARKER"
        printf '  The marker appears only when the runner exits. If the group\n'
        printf '  below is at target size and campaign.out is still growing,\n'
        printf '  the campaign is working.\n'
    else
        state="$(printf '%s\n' "$body" | awk '$1 == "state" { print $2; exit }')"
        teardown="$(printf '%s\n' "$body" | awk '$1 == "teardown" { print $2; exit }')"
        warning="$(printf '%s\n' "$body" | sed -n 's/^warning  *//p')"
        case "$state" in
            DONE)
                printf 'STATE: FINISHED -- the campaign completed and its final sync passed\n' ;;
            FAILED)
                printf 'STATE: FAILED -- the runner exited non-zero; read campaign.out below\n' ;;
            *)
                printf 'STATE: UNREADABLE -- %s exists but declares no state\n' "$MARKER" ;;
        esac
        printf '%s\n' "$body" | sed 's/^/  /'
        case "$teardown" in
            resized)
                printf '\n  The machine took itself down: the group is at size 0 and the VM\n'
                printf "  is gone. Group and template survive, cost nothing, and are what\n"
                printf "  'down' removes.\n" ;;
            resize-requested)
                printf '\n  A self-teardown was in flight when this marker was written. The\n'
                printf '  group below settles it: targetSize 0 means it landed.\n' ;;
            poweroff)
                printf '\n  No managed group owned that instance; it powered itself off.\n' ;;
        esac
        if [ -n "$warning" ]; then
            printf '\n'
            printf '  *** ACTION REQUIRED -- A MACHINE IS STILL BILLING ***\n'
            printf '  *** %s\n' "$warning"
            printf '  *** Collect and destroy it by hand:\n'
            printf '  ***     cloud/campaign.sh fetch\n'
            printf '  ***     cloud/campaign.sh down\n'
        fi
    fi

    # Anything else under status/: the alert objects, and the markers of the
    # other shards of this block when the work was split across machines.
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        case "$line" in
            *"/$SHARD.txt") continue ;;
            */ACTION_REQUIRED_*)
                printf '\n  *** %s\n' "$line"
                gcloud storage cat "$line" 2>/dev/null | sed 's/^/  *** /' || true
                continue ;;
        esac
        printf '  other shard %-28s %s\n' "$(basename "$line")" \
            "$(gcloud storage cat "$line" 2>/dev/null \
                | awk '$1 == "state" { print $2; exit }')"
    done < <(gcloud storage ls "$GCS/status/" 2>/dev/null || true)
}

cmd_status() {
    require_gcloud
    rule "campaign state"
    report_state
    rule "group $MIG_NAME"
    if mig_exists; then
        gcloud compute instance-groups managed describe "$MIG_NAME" \
            --project="$PQL_PROJECT" --zone="$PQL_ZONE" \
            --format='value[separator="  "](name, targetSize, status.isStable, instanceTemplate.basename())'
        gcloud compute instance-groups managed list-instances "$MIG_NAME" \
            --project="$PQL_PROJECT" --zone="$PQL_ZONE" \
            --format='table(name, status, currentAction, instanceStatus)'
        rule "recent group errors"
        gcloud compute instance-groups managed list-errors "$MIG_NAME" \
            --project="$PQL_PROJECT" --zone="$PQL_ZONE" \
            --format='table(error.code, error.message.slice(0:70), timestamp)' \
            2>/dev/null | head -8 || true
    else
        note "no managed group $MIG_NAME in $PQL_ZONE"
    fi

    rule "bucket $GCS"
    local archives
    archives="$(gcloud storage ls --recursive "$GCS/results/**" 2>/dev/null \
        | grep -c '\.npz$' || true)"
    printf 'archives in the bucket: %s\n' "${archives:-0}"
    printf 'hosts seen: %s\n' \
        "$(gcloud storage ls "$GCS/logs/" 2>/dev/null | wc -l | tr -d ' ')"

    local host
    host="$(default_host)"
    if [ -n "$host" ]; then
        rule "campaign.out ($host, last 20 lines)"
        gcloud storage cat "$GCS/logs/$host/campaign.out" 2>/dev/null \
            | tail -20 || true
    fi

    rule "last fetch"
    if [ -f "$FETCH_STAMP" ]; then cat "$FETCH_STAMP"; else printf 'never\n'; fi
}

cmd_logs() {
    require_gcloud
    local follow=0 host=""
    while [ $# -gt 0 ]; do
        case "$1" in
            -f|--follow) follow=1 ;;
            --host) host="${2:-}"; shift ;;
            *) die "logs: unknown option '$1'" ;;
        esac
        if [ $# -gt 0 ]; then shift; fi
    done
    [ -n "$host" ] || host="$(default_host)"
    [ -n "$host" ] || die "no host has written logs to $GCS/logs yet"
    local object="$GCS/logs/$host/campaign.out"

    if [ "$follow" -eq 0 ]; then
        gcloud storage cat "$object"
        return
    fi

    note "following $object (Ctrl-C to stop)"
    local buffer offset=0 size
    buffer="$(mktemp)"
    trap 'rm -f "$buffer"' EXIT
    while true; do
        if gcloud storage cat "$object" > "$buffer" 2>/dev/null; then
            size="$(wc -c < "$buffer" | tr -d ' ')"
            if [ "$size" -lt "$offset" ]; then offset=0; fi
            if [ "$size" -gt "$offset" ]; then
                tail -c "+$((offset + 1))" "$buffer"
                offset="$size"
            fi
        fi
        sleep "$PQL_FOLLOW_INTERVAL"
    done
}

cmd_fetch() {
    require_gcloud
    mkdir -p "$PQL_LOCAL_STATE" "$PQL_LOCAL_LOGS" "$REPO_DIR/results"

    local leg status=ok
    for leg in results analysis logs; do
        gcloud storage ls "$GCS/$leg" >/dev/null 2>&1 || { note "no $leg in the bucket"; continue; }
        case "$leg" in
            results)  note "results  -> $REPO_DIR/results"
                      gcloud storage rsync --recursive "$GCS/results" "$REPO_DIR/results" ;;
            analysis) note "analysis -> $REPO_DIR/analysis"
                      gcloud storage rsync --recursive "$GCS/analysis" "$REPO_DIR/analysis"
                      # The VMs push the *.log files of their pinned clone,
                      # which are the committed versions of an older commit:
                      # this leg exists for logs the campaign itself wrote,
                      # but it also stomps any log regenerated locally since
                      # the pin. Committed logs are the record, so anything
                      # this rsync just made differ from git is restored,
                      # and genuinely new logs (untracked) are kept.
                      if command -v git >/dev/null 2>&1 \
                          && git -C "$REPO_DIR" rev-parse >/dev/null 2>&1; then
                          local stomped
                          stomped="$(git -C "$REPO_DIR" diff --name-only \
                              -- 'analysis/*.log' 2>/dev/null)"
                          if [ -n "$stomped" ]; then
                              note "restoring committed logs the fetch overwrote:"
                              printf '%s\n' "$stomped" | sed 's/^/      /'
                              (cd "$REPO_DIR" \
                               && git checkout -- $stomped) || true
                          fi
                      fi ;;
            logs)     note "logs     -> $PQL_LOCAL_LOGS"
                      gcloud storage rsync --recursive "$GCS/logs" "$PQL_LOCAL_LOGS" ;;
        esac
    done

    rule "integrity gate"
    local python
    python="$(pick_python)"
    if ! "$python" -c 'import numpy' >/dev/null 2>&1; then
        status=gate-unavailable
        note "no interpreter with numpy found (tried $python); set PQL_PYTHON"
        note "the mirror is complete, the gate simply could not run"
    elif ( cd "$REPO_DIR" && "$python" pq_validate.py --allow-missing ); then
        note "integrity gate passed"
    else
        status=validate-failed
        note "integrity gate FAILED: the files are home, the corpus is not"
        note "expect this while a block is mid-flight -- the gate asserts the"
        note "archived file count per block, which a partial campaign exceeds"
        note "or undershoots. Read the failures before trusting the mirror."
    fi

    {
        printf 'status  %s\n' "$status"
        printf 'when    %s\n' "$(date -u +%FT%TZ)"
        printf 'source  %s\n' "$GCS"
        printf 'results %s\n' "$REPO_DIR/results"
        printf 'logs    %s\n' "$PQL_LOCAL_LOGS"
        printf 'python  %s\n' "$python"
    } > "$FETCH_STAMP"
    note "fetch recorded in $FETCH_STAMP"
    [ "$status" = ok ]
}

# Refuse to tear down over unfetched work. The stamp says a fetch happened;
# the dry run says nothing has landed in the bucket since -- a campaign that
# is still running writes for as long as it lives, and the fetch that ran ten
# minutes ago is already behind it.
bucket_has_unfetched_work() {
    local pending=""
    gcloud storage rsync --help 2>/dev/null | grep -q -- '--dry-run' || return 1
    if gcloud storage ls "$GCS/results" >/dev/null 2>&1; then
        pending="$(gcloud storage rsync --recursive --dry-run \
            "$GCS/results" "$REPO_DIR/results" 2>/dev/null | head -5)"
    fi
    if [ -z "$pending" ] && gcloud storage ls "$GCS/analysis" >/dev/null 2>&1; then
        pending="$(gcloud storage rsync --recursive --dry-run \
            "$GCS/analysis" "$REPO_DIR/analysis" 2>/dev/null | head -5)"
    fi
    [ -n "$pending" ]
}

cmd_down() {
    require_gcloud
    local force=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --force) force=1 ;;
            *) die "down: unknown option '$1'" ;;
        esac
        shift
    done

    if [ ! -f "$FETCH_STAMP" ]; then
        die "refusing to tear down: no fetch has ever succeeded for $MIG_NAME.
     The bucket outlives the VM, but the VM is where the campaign is; run
     'cloud/campaign.sh fetch' first. There is no --force for this case."
    fi
    local status
    status="$(awk '$1 == "status" { print $2 }' "$FETCH_STAMP")"
    if [ "$status" != "ok" ] && [ "$force" -ne 1 ]; then
        die "refusing to tear down: the last fetch ended '$status'
     ($(awk '$1 == "when" { print $2 }' "$FETCH_STAMP")).
     Fix the integrity gate, or re-run with --force if the mirror itself is
     complete and only the gate is unhappy."
    fi
    if bucket_has_unfetched_work; then
        if [ "$force" -ne 1 ]; then
            die "refusing to tear down: the bucket holds objects this machine
     does not. Run 'cloud/campaign.sh fetch' again. If the campaign is still
     writing and you mean to stop it anyway, 'down --force' will take it
     down and leave those objects in the bucket for a later fetch."
        fi
        note "WARNING: the bucket holds objects this machine does not."
        note "Nothing is destroyed -- deleting the group deletes the VM, not"
        note "the bucket -- but run 'fetch' again afterwards to collect them."
    fi

    if ! mig_exists; then
        note "no managed group $MIG_NAME; nothing to delete"
        return 0
    fi
    local template
    template="$(basename "$(gcloud compute instance-groups managed describe \
        "$MIG_NAME" --project="$PQL_PROJECT" --zone="$PQL_ZONE" \
        --format='value(instanceTemplate)')")"

    note "deleting group $MIG_NAME (this stops the recreation loop)"
    gcloud compute instance-groups managed delete "$MIG_NAME" \
        --project="$PQL_PROJECT" --zone="$PQL_ZONE" --quiet

    if [ -n "$template" ]; then
        note "deleting template $template"
        gcloud compute instance-templates delete "$template" \
            --project="$PQL_PROJECT" --quiet
    fi
    note "down. The bucket $GCS is untouched and still holds everything."
}

case "${1:-}" in
    up)     shift; cmd_up "$@" ;;
    status) shift; cmd_status "$@" ;;
    logs)   shift; cmd_logs "$@" ;;
    fetch)  shift; cmd_fetch "$@" ;;
    down)   shift; cmd_down "$@" ;;
    -h|--help|help|"") usage ;;
    *) usage >&2; die "unknown command '$1'" ;;
esac
