#!/usr/bin/env bash
set -euo pipefail
run_root=/root/autodl-tmp
pid_file="$run_root/blockbox-ab-baseline.pid"
if [[ -f "$pid_file" ]]; then
  prior_pid=$(cat "$pid_file")
  if [[ "$prior_pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$prior_pid/cmdline" ]] &&
     tr '\0' ' ' < "/proc/$prior_pid/cmdline" | grep -Fq '/root/autodl-tmp/serve_blockbox_ab.py'; then
    echo "Baseline server already running: $prior_pid"
    exit 0
  fi
fi
# Free the superseded synchronous service, leaving the current RTC service intact.
for prior_pid in $(pgrep -f '^/root/autodl-tmp/pi05-run/.venv/bin/python -u /root/autodl-tmp/serve_blockbox_remote.py .* --port 8091$' || true); do
  kill -TERM "$prior_pid"
  for attempt in {1..20}; do
    if ! kill -0 "$prior_pid" 2>/dev/null; then break; fi
    sleep 0.5
  done
  if kill -0 "$prior_pid" 2>/dev/null; then echo 'Sync service did not exit'; exit 1; fi
done
export PYTHONPATH="$run_root/lerobot-044-src/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
nohup "$run_root/pi05-run/.venv/bin/python" -u "$run_root/serve_blockbox_ab.py" --port 8093 \
  > "$run_root/blockbox-ab-baseline.log" 2>&1 < /dev/null &
echo $! > "$pid_file"
echo "BASELINE_LOADING PID $(cat "$pid_file")"
