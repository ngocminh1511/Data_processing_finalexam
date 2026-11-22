#!/usr/bin/env bash
# Chạy một bước, bấm giờ, lưu log.
#   bash scripts/run_step.sh <tên_bước> <lệnh...>
#   ví dụ: bash scripts/run_step.sh select_uniform_50 modal run select_frames.py --method uniform --limit 50
# Log đầy đủ: outputs/logs/<tên_bước>.log
# Thời gian:  outputs/logs/timing.tsv  (step, start, end, seconds, exit_code, command)
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
[ -d .venv/Scripts ] && export PATH="$PWD/.venv/Scripts:$PATH"
[ -d .venv/bin ] && export PATH="$PWD/.venv/bin:$PATH"

name=$1; shift
mkdir -p outputs/logs
log="outputs/logs/$name.log"
tsv="outputs/logs/timing.tsv"
[ -f "$tsv" ] || printf "step\tstart\tend\tseconds\texit_code\tcommand\n" > "$tsv"

start=$(date '+%Y-%m-%d %H:%M:%S'); t0=$(date +%s)
echo "[$start] $*" | tee "$log"
"$@" 2>&1 | tee -a "$log"
code=${PIPESTATUS[0]}
end=$(date '+%Y-%m-%d %H:%M:%S'); sec=$(( $(date +%s) - t0 ))
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$start" "$end" "$sec" "$code" "$*" >> "$tsv"
echo "[$end] $name: ${sec}s, exit=$code" | tee -a "$log"
exit "$code"
