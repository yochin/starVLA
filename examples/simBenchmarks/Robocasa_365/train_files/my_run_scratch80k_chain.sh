#!/bin/bash
# QwenOFT 동등 비교용 GR00T scratch 실험 2개를 순차 실행한다.
#   A: starvla_qwengr00t_g1_dexhands_scratch80k          (state 사용, 81D)
#   B: starvla_qwengr00t_g1_dexhands_scratch80k_nostate  (state 미사용)
#
# H200 1장이라 동시 실행이 불가능하므로 순차로 돌린다. A의 평가 단계가 실패해도
# B 학습은 계속되어야 하므로 set -e 는 쓰지 않고, 각 단계의 종료코드만 기록한다.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LOG_DIR="$REPO_ROOT/results/Checkpoints/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_A="$LOG_DIR/scratch80k_${STAMP}.log"
LOG_B="$LOG_DIR/scratch80k_nostate_${STAMP}.log"

echo "[chain] 시작: $(date)"
echo "[chain] A 로그: $LOG_A"
echo "[chain] B 로그: $LOG_B"

echo "[chain] === A 시작 (state 사용): $(date) ==="
bash "$SCRIPT_DIR/my_run_robocasa365_scratch80k.sh" > "$LOG_A" 2>&1
RC_A=$?
echo "[chain] === A 종료 (exit=$RC_A): $(date) ==="

echo "[chain] === B 시작 (state 미사용): $(date) ==="
bash "$SCRIPT_DIR/my_run_robocasa365_scratch80k_nostate.sh" > "$LOG_B" 2>&1
RC_B=$?
echo "[chain] === B 종료 (exit=$RC_B): $(date) ==="

echo "[chain] 완료: A exit=$RC_A, B exit=$RC_B"
exit $(( RC_A != 0 || RC_B != 0 ))
