#!/bin/bash
set -e

# G1 냉장고 5개 태스크(Apple / GraspLast / Onion / PickCoke / TakeCoke) QwenOFT 학습.
#
# 모델 선택 근거 (grapes 2개 데이터셋 기준 동등 비교 결과):
#   - 정확도는 QwenOFT 와 GR00T-nostate 가 통계적으로 동률 (페어드 부트스트랩 95% CI 가 0 포함)
#   - 추론 지연시간은 QwenOFT 48.7ms(20.6Hz) vs GR00T-nostate 69.4ms(14.4Hz) 로 1.4배 빠름
#   - 학습 시간도 19.0h vs 21.7h 로 저렴
#   → 동률이면 싼 쪽을 고른다. 실로봇 제어 주기에서도 여유가 크다.
#
# head(dims 0,1) 는 이번에는 학습한다. 데이터상 head 는 팔 관절과 비슷한 진폭으로
# 움직이고(std 0.11~0.35 rad, 범위 ±1.0~1.4), head 는 front 카메라 방향을 정하므로
# 예측이 망가지면 관측 자체가 어긋난다. 마스킹한 과거 run 들은 head MSE 0.31~0.39,
# 학습했던 stage1/2 는 0.062/0.066 으로 5배 좋았다.
# 손 이산 모드(dims 10,36)만 계속 제외한다 — 값이 {0,1,2,3} 이라 L1 회귀 대상이 아니다.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHON="/home/yochin/miniforge3/envs/starVLA/bin/python"
export ACCELERATE="/home/yochin/miniforge3/envs/starVLA/bin/accelerate"
export MUJOCO_GL=egl

CONFIG_YAML="examples/simBenchmarks/Robocasa_365/train_files/starvla_qwenoft_g1_fridge5.yaml"
RUN_ID="starvla_qwenoft_g1_fridge5"
RUN_DIR="$REPO_ROOT/results/Checkpoints/$RUN_ID"

# 데이터 복사가 끝났는지 먼저 확인한다. 복사 중에 시작하면 통계와 스텝 인덱스가
# 불완전한 상태로 캐시되어 학습 내내 잘못된 정규화를 쓰게 된다.
echo ">> 데이터셋 확인"
MISSING=0
for d in FridgeApple FridgeGraspLast FridgeOnion FridgePickCoke FridgeTakeCoke; do
    for split in train val; do
        p="$REPO_ROOT/playground/Datasets/$d/$split"
        if [ ! -f "$p/meta/info.json" ]; then
            echo "   [없음] $d/$split"; MISSING=1; continue
        fi
        n_ep=$($PYTHON -c "import json;print(json.load(open('$p/meta/info.json'))['total_episodes'])")
        n_pq=$(find "$p/data" -name '*.parquet' 2>/dev/null | wc -l)
        n_vid=$(find "$p/videos" -name '*.mp4' 2>/dev/null | wc -l)
        if [ "$n_ep" -ne "$n_pq" ] || [ "$n_vid" -ne $((n_ep * 3)) ]; then
            echo "   [불완전] $d/$split  에피소드 $n_ep / parquet $n_pq / 비디오 $n_vid (기대 $((n_ep * 3)))"
            MISSING=1
        else
            echo "   [정상]   $d/$split  에피소드 $n_ep"
        fi
    done
done
if [ "$MISSING" -ne 0 ]; then
    echo "!! 데이터 복사가 끝나지 않았습니다. 완료 후 다시 실행하세요."
    exit 1
fi

echo "=========================================================="
echo " G1 Fridge 5태스크 QwenOFT 학습 시작"
echo "  설정: $CONFIG_YAML"
echo "  Loss 제외: 손 이산 모드 (dims 10, 36) 만 — head(0,1) 는 학습함"
echo "  State 입력: 미사용 (include_state=false)"
echo "=========================================================="

WANDB_MODE=online \
TOKENIZERS_PARALLELISM=false \
$ACCELERATE launch \
    --config_file "$REPO_ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
    --num_processes 1 \
    "$REPO_ROOT/starVLA/training/train_starvla.py" \
    --config_yaml "$REPO_ROOT/$CONFIG_YAML"

echo "=========================================================="
echo " 학습 완료. 평가 시작: $RUN_DIR"
echo "=========================================================="

CKPT="$RUN_DIR/final_model/pytorch_model.pt"

# 평가 집합을 이 5개로 못박는다. 지정하지 않으면 openloop_mse.py 는 기본값
# (grapes 2개 + Apple) 을 쓰므로 학습한 태스크 대부분이 평가에서 빠진다.
export G1_EVAL_DATASETS="FridgeApple,FridgeGraspLast,FridgeOnion,FridgePickCoke,FridgeTakeCoke"
echo ">> 평가 대상: $G1_EVAL_DATASETS"

# QwenOFT 는 state 없이 학습했으므로 평가에서도 state 를 보내지 않는다.
echo ">> 1. 오픈루프 MSE (openloop_mse.py)"
$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/openloop_mse.py" \
    --trained_ckpt "$CKPT" --split both --no_send_state \
    --out "$RUN_DIR/openloop_mse.json"

echo ">> 2. 궤적 플롯 및 Horizon Drift (plot_openloop_traj.py)"
$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/plot_openloop_traj.py" \
    --ckpt "$CKPT" --split both --no_send_state \
    --num_chunks 60 --out_dir "$RUN_DIR/traj_plots"

echo ">> 3. 관절별 오차 및 손끝 위치 오차 (perjoint_mse.py & ee_error_attrib.py)"
$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/perjoint_mse.py" \
    --ckpt "$CKPT" --split both --no_send_state \
    --out "$RUN_DIR/perjoint.json"

$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/ee_error_attrib.py" \
    --npz "$RUN_DIR/perjoint.npz" --hand right \
    --out "$RUN_DIR/ee_attrib_right.json"

echo ">> 4. MuJoCo 3패널 동기 비디오 (dump_pred_traj.py & render_replay_sync.py)"
$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/dump_pred_traj.py" \
    --ckpt "$CKPT" --split both --no_send_state \
    --num_chunks 60 --episodes_per_task 1 \
    --out "$RUN_DIR/pred_traj_full.npz"

$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/render_replay_sync.py" \
    --npz "$RUN_DIR/pred_traj_full.npz" \
    --out_dir "$RUN_DIR/replay_videos_sync"

echo "=========================================================="
echo " ✅ 학습 및 평가 완료. 결과: $RUN_DIR"
echo "=========================================================="
