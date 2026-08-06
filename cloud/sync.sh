#!/usr/bin/env bash
# Push everything the campaign has produced to GCS, every two minutes.
#
# This is the file whose absence lost the logs of the last campaign: the VM
# was torn down and its run logs went with it. A spot machine is disposable
# by construction, so nothing that is only on its disk exists. The loop below
# makes the bucket the record and the disk a cache.
#
# What goes where:
#     results/         -> GCS/results       shared: archives, per-job logs and
#                                           the restart cache all live here
#     analysis/*.log   -> GCS/analysis      shared: these logs ARE artifacts
#     campaign.out     -> GCS/logs/<host>/  per host
#
# The split is not cosmetic. The shared legs are job artifacts: a recreated
# machine restores them and the runner then skips the work they represent.
# campaign.out is per-machine narration, and two machines sharding one block
# would otherwise overwrite each other's copy of the same object name.
#
# *.partial files are excluded, and that exclusion is load-bearing twice
# over. The runner stages a log-artifact job as <name>.log.partial and
# renames it into place only on success, so a partial in the bucket would
# become a skip marker for work that failed; and the restart cache stages
# each entry under a fixed .partial name, which is meaningless to anyone but
# the writing process.
#
# Usage:
#     cloud/sync.sh            # loop until killed (started by startup.sh)
#     cloud/sync.sh --once     # one pass, for a manual push
set -euo pipefail

ENV_FILE="${PQL_ENV_FILE:-/etc/pql/campaign.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$ENV_FILE"
fi

REPO="${PQL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GCS="${PQL_GCS:-${PQL_BUCKET:-}/${PQL_PREFIX:-converged}}"
HOST="${PQL_HOST:-$(hostname)}"
RESTART_CACHE="${PQL_RESTART_CACHE:-results/restart_cache}"
INTERVAL="${PQL_SYNC_INTERVAL:-120}"
STATE_DIR="${PQL_STATE_DIR:-/var/lib/pql}"
SKIP='.*\.partial$'

ONCE=0
if [ "${1:-}" = "--once" ]; then ONCE=1; fi

case "$GCS" in
    gs://*) ;;
    *) printf 'sync: no bucket configured (PQL_BUCKET or %s)\n' "$ENV_FILE" >&2
       exit 1 ;;
esac

mkdir -p "$STATE_DIR"
cd "$REPO" || { printf 'sync: cannot enter %s\n' "$REPO" >&2; exit 1; }

log() { printf '[pql-sync %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# One probe, once: older Cloud CLI builds ship `gcloud storage rsync` without
# --exclude, and silently uploading the partials would be worse than failing.
if gcloud storage rsync --help 2>/dev/null | grep -q -- '--exclude'; then
    MIRROR_TOOL=gcloud
elif command -v gsutil >/dev/null 2>&1; then
    MIRROR_TOOL=gsutil
else
    printf 'sync: no rsync tool with exclusion support (gcloud storage / gsutil)\n' >&2
    exit 1
fi

mirror() {
    local src="$1" dst="$2" exclude="$3"
    [ -d "$src" ] || return 0
    if [ "$MIRROR_TOOL" = gcloud ]; then
        gcloud storage rsync --recursive --exclude="$exclude" "$src" "$dst"
    else
        gsutil -m rsync -r -x "$exclude" "$src" "$dst"
    fi
}

push_file() {
    local src="$1" dst="$2"
    [ -f "$src" ] || return 0
    gcloud storage cp "$src" "$dst"
}

cycle() {
    mirror "$REPO/results" "$GCS/results" "$SKIP"
    # The restart cache lives under results/ by default and is already
    # covered; this leg only fires if it is ever moved out of the tree.
    case "$RESTART_CACHE" in
        results/*) ;;
        *) mirror "$REPO/$RESTART_CACHE" "$GCS/restart-cache" "$SKIP" ;;
    esac

    local log_file
    while IFS= read -r log_file; do
        push_file "$log_file" "$GCS/analysis/$(basename "$log_file")"
    done < <(find "$REPO/analysis" -maxdepth 1 -name '*.log' -type f 2>/dev/null)

    push_file "$REPO/campaign.out" "$GCS/logs/$HOST/campaign.out"
}

RUNNING=1
trap 'RUNNING=0' TERM INT

log "mirroring $REPO -> $GCS every ${INTERVAL}s via $MIRROR_TOOL"
while [ "$RUNNING" -eq 1 ]; do
    # The stamp is taken BEFORE the pass, not after: a file written while the
    # pass is running has not been pushed, and shutdown.sh must still see it
    # as new. Better a file sent twice than a file lost.
    started="$STATE_DIR/.sync-started"
    : > "$started"
    begin="$SECONDS"
    if cycle >"$STATE_DIR/sync-last.out" 2>&1; then
        mv -f "$started" "$STATE_DIR/last-sync"
        log "pass ok in $((SECONDS - begin))s"
    else
        rm -f "$started"
        log "pass FAILED after $((SECONDS - begin))s; retrying next cycle"
        tail -5 "$STATE_DIR/sync-last.out" >&2 || true
    fi
    if [ "$ONCE" -eq 1 ]; then break; fi
    sleep "$INTERVAL"
done
