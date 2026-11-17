#!/usr/bin/env bash
set -euo pipefail
run_root=/root/autodl-tmp
pid_file="$run_root/blockbox-rtc-server-020000.pid"
if [[ -f "$pid_file" ]]; then
  prior_pid=$(cat "$pid_file")
  if [[ "$prior_pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$prior_pid/cmdline" ]] &&
     tr '\0' ' ' < "/proc/$prior_pid/cmdline" | grep -Fq '/root/autodl-tmp/serve_blockbox_rtc.py'; then
    echo "RTC server process already running: $prior_pid (readiness: http://127.0.0.1:8092/health)"
    exit 0
  fi
fi
export PYTHONPATH="$run_root/lerobot-044-src/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
nohup "$run_root/pi05-run/.venv/bin/python" -u "$run_root/serve_blockbox_rtc.py" --port 8092 \
  > "$run_root/blockbox-rtc-server-020000.log" 2>&1 < /dev/null &
echo $! > "$pid_file"
echo "RTC server loading, PID $(cat "$pid_file"); inspect $run_root/blockbox-rtc-server-020000.log for RTC_READY"
