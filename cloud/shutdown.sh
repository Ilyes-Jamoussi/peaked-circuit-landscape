#!/usr/bin/env bash
# Last words before the machine disappears.
#
# GCP sends the ACPI signal about 30 seconds before it reclaims a spot
# instance, and the disk is deleted with it (the MIG needs
# --instance-termination-action=DELETE to be able to recreate anything). So
# this script has one job: get the files written since the last successful
# sync pass into the bucket, newest and smallest first, and stop when the
# clock runs out rather than dying mid-upload.
#
# It does NOT re-mirror the tree. A full pass over results plus the restart
# cache takes minutes; a scan for files newer than the sync stamp takes
# milliseconds and is all that can be honestly attempted in the window.
#
# Priority order, each phase capped by what remains of the budget:
#     1. the termination marker and campaign.out                (kilobytes)
#     2. analysis logs, then everything new under results/      (megabytes)
#        -- which is where the per-job logs and the restart cache live too
#
# Residual risk, stated rather than hidden: an .npz being written at the
# instant of preemption can be copied up truncated, and a truncated archive
# would look like a finished job to the runner. Files touched in the last few
# seconds are skipped for that reason, and pq_validate.py -- which loads
# every archive and checks its shapes -- is the backstop at fetch time.
#
# Usage: installed as the instance-template shutdown-script metadata by
# cloud/provision.sh. Never run by hand.
set -euo pipefail

ENV_FILE="${PQL_ENV_FILE:-/etc/pql/campaign.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE"
fi

# A preemption during the first boot arrives before startup.sh has written
# that file. Read the instance metadata instead, so that even the boot that
# never got started leaves a record of why it stopped.
if [ -z "${PQL_BUCKET:-}" ]; then
    meta() {
        curl -fsS -H 'Metadata-Flavor: Google' \
            "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
            2>/dev/null || true
    }
    PQL_BUCKET="$(meta pql-bucket)"
    PQL_PREFIX="${PQL_PREFIX:-$(meta pql-prefix)}"
    PQL_BLOCK="${PQL_BLOCK:-$(meta pql-block)}"
fi

REPO="${PQL_REPO:-/opt/pql/peaked-circuit-landscape}"
GCS="${PQL_GCS:-${PQL_BUCKET:-}/${PQL_PREFIX:-converged}}"
HOST="${PQL_HOST:-$(hostname)}"
RESTART_CACHE="${PQL_RESTART_CACHE:-results/restart_cache}"
STATE_DIR="${PQL_STATE_DIR:-/var/lib/pql}"
DEADLINE="${PQL_SHUTDOWN_DEADLINE:-25}"
PARALLEL="${PQL_SHUTDOWN_PARALLEL:-8}"
SETTLE="${PQL_SHUTDOWN_SETTLE:-5}"   # seconds; skip files still being written

log() { printf '[pql-shutdown %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

case "$GCS" in
    gs://*) ;;
    *) log "no bucket configured; nothing to save"; exit 0 ;;
esac
[ -d "$REPO" ] || { log "no repository at $REPO"; exit 0; }

END=$(( $(date +%s) + DEADLINE ))
left() {
    local now
    now="$(date +%s)"
    if [ "$END" -gt "$now" ]; then printf '%s' $((END - now)); else printf '0'; fi
}

cd "$REPO" || { log "cannot enter $REPO"; exit 0; }

STAMP="$STATE_DIR/last-sync"
if [ -f "$STAMP" ]; then
    NEWER=(-newer "$STAMP")
else
    NEWER=(-newermt "@0")
    log "no sync stamp: every file counts as new, and they will not all fit"
fi
SETTLE_TS=$(( $(date +%s) - SETTLE ))

# Copy a NUL-separated list of paths, each relative to the current directory,
# under a destination root that mirrors the same relative layout.
push_list() {
    local dest_root="$1" budget="$2"
    [ "$budget" -gt 1 ] || return 0
    timeout "$budget" xargs -0 -r -P "$PARALLEL" -I@ \
        gcloud storage cp "@" "$dest_root/@" || true
}

# ------------------------------------------------- 1. marker and narration
PREEMPTED="$(curl -fsS -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/preempted' \
    2>/dev/null || printf 'UNKNOWN')"
STAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
{
    printf 'host       %s\n' "$HOST"
    printf 'stopped    %s\n' "$(date -u +%FT%TZ)"
    printf 'preempted  %s\n' "$PREEMPTED"
    printf 'block      %s\n' "${PQL_BLOCK:-unknown}"
    printf 'last sync  %s\n' \
        "$([ -f "$STAMP" ] && date -u -r "$STAMP" +%FT%TZ || printf 'never')"
    printf 'campaign   %s\n' "$(systemctl is-active pql-campaign 2>/dev/null || true)"
} | timeout 5 gcloud storage cp - "$GCS/logs/$HOST/TERMINATED_$STAMP_UTC.txt" \
    >/dev/null 2>&1 || true

# Every phase ends in `|| true`. Under `set -e` with `pipefail` a find that
# trips over one unreadable path would abort the script and take the phases
# after it down too, which in a 30-second window means losing the archives to
# save nothing. Best effort here beats correctness-or-nothing.
if [ -f campaign.out ]; then
    printf '%s\0' campaign.out \
        | push_list "$GCS/logs/$HOST" "$(left)" >/dev/null 2>&1 || true
fi
log "narration pushed, $(left)s left"

# ------------------------------------------------ 2. analysis logs, results
find analysis -maxdepth 1 -name '*.log' -type f "${NEWER[@]}" -print0 2>/dev/null \
    | push_list "$GCS" "$(left)" >/dev/null 2>&1 || true

find results -type f "${NEWER[@]}" ! -newermt "@$SETTLE_TS" \
    ! -name '*.partial' -print0 2>/dev/null \
    | push_list "$GCS" "$(left)" >/dev/null 2>&1 || true
log "results pushed, $(left)s left"

# --------------------------------- 3. the restart cache, if it moved outside
if [ -d "$RESTART_CACHE" ] && [ "$(left)" -gt 2 ]; then
    case "$RESTART_CACHE" in
        results/*) ;;
        *) ( cd "$RESTART_CACHE" \
                && find . -type f "${NEWER[@]}" ! -newermt "@$SETTLE_TS" \
                    ! -name '*.partial' -print0 2>/dev/null \
                | push_list "$GCS/restart-cache" "$(left)" ) >/dev/null 2>&1 || true ;;
    esac
fi

log "done with $(left)s to spare"
