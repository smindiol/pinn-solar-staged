#!/bin/bash
# Launch one training job on the shared workstation GPU.
#
#   gpu_launch.sh <session-name> <command...>
#   gpu_launch.sh --force <session-name> <command...>   # skip the busy check
#
# 1. Refuses to start when someone else runs a GPU job (any compute process
#    that is neither ours nor desktop/remote-desktop software). Our own job
#    may use the whole GPU; there is no utilization or memory cap.
# 2. Starts the command inside tmux, logging to ~/smindiola/runs/<session>.log.
# 3. Opens a full-screen, read-only terminal on the workstation desktop
#    showing the job, so other users can see the GPU is in use.
set -euo pipefail

# Desktop and remote-desktop processes that always hold a little GPU memory.
IGNORE='^(rustdesk|gnome-shell|Xwayland|Xorg|gnome-remote-desktop|firefox|chrome)$'
FORCE=0
if [ "${1:-}" = "--force" ]; then FORCE=1; shift; fi
[ $# -ge 2 ] || { echo "usage: $0 [--force] <session> <command...>"; exit 2; }
SESSION=$1; shift
CMD="$*"
RUNS=$HOME/smindiola/runs
LOG=$RUNS/$SESSION.log
mkdir -p "$RUNS"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[gpu_launch] session $SESSION already exists; refusing"; exit 3
fi

# PIDs that belong to our own tmux jobs (and their children).
ours() {
    for p in $(tmux list-panes -a -F '#{pane_pid}' 2>/dev/null); do
        echo "$p"; pgrep -P "$p" -d ' ' 2>/dev/null | tr ' ' '\n'
        for c in $(pgrep -P "$p" 2>/dev/null); do pgrep -P "$c" 2>/dev/null; done
    done | sort -u
}

if [ $FORCE -eq 0 ]; then
    OURS=$(ours)
    FOREIGN_MB=0
    FOREIGN_JOBS=0
    while IFS=', ' read -r pid mem; do
        [ -z "$pid" ] && continue
        if ! grep -qx "$pid" <<<"$OURS"; then
            comm=$(ps -o comm= -p "$pid" 2>/dev/null || echo unknown)
            if grep -qE "$IGNORE" <<<"$comm"; then
                echo "[gpu_launch] ignoring desktop process pid=$pid $comm ${mem}MiB"
                continue
            fi
            FOREIGN_JOBS=$((FOREIGN_JOBS + 1))
            FOREIGN_MB=$((FOREIGN_MB + mem))
            echo "[gpu_launch] FOREIGN GPU job pid=$pid ${mem}MiB $(ps -o user=,etime=,args= -p "$pid" 2>/dev/null | cut -c1-120)"
        fi
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
    echo "[gpu_launch] foreign jobs=${FOREIGN_JOBS} (${FOREIGN_MB}MiB)"
    if [ "$FOREIGN_JOBS" -gt 0 ]; then
        echo "[gpu_launch] someone else is running a GPU job; not starting (use --force only with the owner's OK)"
        exit 4
    fi
fi

tmux new -d -s "$SESSION" "bash -c '$CMD' 2>&1 | tee $LOG; echo; echo '[job finished]'; bash"
echo "[gpu_launch] started tmux session $SESSION, log $LOG"

# Full-screen, read-only monitor on the workstation desktop.
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
export DISPLAY=${DISPLAY:-:0}
BANNER="echo; echo '   ENTRENAMIENTO PINN EN CURSO EN LA GPU - POR FAVOR NO CERRAR NI APAGAR'; \
echo '   Responsable: smindiola  |  sesion: $SESSION  |  cerrar esta ventana NO detiene el trabajo'; echo; sleep 3"
gnome-terminal --full-screen --title="GPU EN USO - $SESSION - NO APAGAR (smindiola)" -- \
    bash -c "$BANNER; tmux attach -r -t $SESSION; exec bash" >/dev/null 2>&1 &
echo "[gpu_launch] monitor window opened on the workstation desktop"
