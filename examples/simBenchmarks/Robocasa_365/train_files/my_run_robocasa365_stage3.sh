#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHON="/home/yochin/miniforge3/envs/starVLA/bin/python"
export ACCELERATE="/home/yochin/miniforge3/envs/starVLA/bin/accelerate"
export MUJOCO_GL=egl

CONFIG_YAML="examples/simBenchmarks/Robocasa_365/train_files/starvla_qwengr00t_g1_dexhands_stage3.yaml"
RUN_ID="starvla_qwengr00t_g1_dexhands_fridge_pick_grapes_stage3"
RUN_DIR="$REPO_ROOT/results/Checkpoints/$RUN_ID"
CKPT="$RUN_DIR/final_model/pytorch_model.pt"

echo "=========================================================="
echo " [Step 1] starVLA Stage 3 모델 학습 시작"
echo "  설정 파일: $CONFIG_YAML"
echo "  기반 체크포인트: results/Checkpoints/starvla_qwengr00t_g1_dexhands_fridge_pick_grapes_stage2/final_model/pytorch_model.pt"
echo "  학습 스텝: 80,000 steps (Save/Eval: 매 20,000 steps)"
echo "  Loss 제외 관절: Head 2D (dims 0, 1) & Hand 이산 모드 (dims 10, 36)"
echo "=========================================================="

WANDB_MODE=online \
TOKENIZERS_PARALLELISM=false \
$ACCELERATE launch \
    --config_file "$REPO_ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
    --num_processes 1 \
    "$REPO_ROOT/starVLA/training/train_starvla.py" \
    --config_yaml "$REPO_ROOT/$CONFIG_YAML"

echo "=========================================================="
echo " [Step 2] Stage 3 학습 완료 후 자동 평가 파이프라인 시작"
echo "  체크포인트: $CKPT"
echo "=========================================================="

# 2-1. 오픈루프 MSE 정량 평가 (Train vs Val & Horizon Step 오차 분석)
echo ">> 2-1. 오픈루프 MSE 정량 평가 (openloop_mse.py)"
$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/openloop_mse.py" \
    --trained_ckpt "$CKPT" \
    --split both \
    --out "$RUN_DIR/openloop_mse.json"

# 2-2. 궤적 플롯 및 Horizon Drift 그래프 생성
echo ">> 2-2. 궤적 플롯 및 Horizon Drift 그래프 생성 (plot_openloop_traj.py)"
$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/plot_openloop_traj.py" \
    --ckpt "$CKPT" \
    --split both \
    --num_chunks 60 \
    --out_dir "$RUN_DIR/traj_plots"

# 2-3. 45개 관절별 오차 & 손끝 위치 오차(cm) 분석
echo ">> 2-3. 관절별 오차 및 손끝 위치 오차 분석 (perjoint_mse.py & ee_error_attrib.py)"
$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/perjoint_mse.py" \
    --ckpt "$CKPT" \
    --split both \
    --out "$RUN_DIR/perjoint.json"

$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/ee_error_attrib.py" \
    --npz "$RUN_DIR/perjoint.npz" \
    --hand right \
    --out "$RUN_DIR/ee_attrib_right.json"

# 2-4. MuJoCo 3패널 동기 비디오 렌더링
echo ">> 2-4. MuJoCo 3패널 동기 비디오 렌더링 (dump_pred_traj.py & render_replay_sync.py)"
$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/dump_pred_traj.py" \
    --ckpt "$CKPT" \
    --split both \
    --num_chunks 60 \
    --episodes_per_task 1 \
    --out "$RUN_DIR/pred_traj_full.npz"

$PYTHON "$REPO_ROOT/examples/G1_Replay/eval_files/render_replay_sync.py" \
    --npz "$RUN_DIR/pred_traj_full.npz" \
    --out_dir "$RUN_DIR/replay_videos_sync"

echo "=========================================================="
echo " ✅ Stage 3 학습 및 모든 평가/렌더링이 성공적으로 완료되었습니다!"
echo " 결과 저장 디렉토리: $RUN_DIR"
echo "=========================================================="
