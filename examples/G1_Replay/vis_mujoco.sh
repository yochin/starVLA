#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHON="/home/yochin/miniforge3/envs/starVLA/bin/python"
export MUJOCO_GL=egl

export CKPT="$REPO_ROOT/results/Checkpoints/starvla_qwengr00t_g1_dexhands_fridge_pick_grapes/final_model/pytorch_model.pt"
export RUN_DIR="$REPO_ROOT/results/Checkpoints/starvla_qwengr00t_g1_dexhands_fridge_pick_grapes"
export TRAJ_NPZ="$RUN_DIR/pred_traj_full.npz"
export VIDEO_OUT_DIR="$RUN_DIR/replay_videos_sync"

echo "=========================================================="
echo "1-1. 연속 open-loop 예측 궤적 덤프 (dump_pred_traj.py)"
echo "=========================================================="
$PYTHON "$SCRIPT_DIR/eval_files/dump_pred_traj.py" \
    --ckpt "$CKPT" \
    --split both \
    --num_chunks 60 \
    --episodes_per_task 1 \
    --out "$TRAJ_NPZ"

echo "=========================================================="
echo "1-2. MuJoCo 3패널 동기 비디오 렌더링 (render_replay_sync.py)"
echo "=========================================================="
$PYTHON "$SCRIPT_DIR/eval_files/render_replay_sync.py" \
    --npz "$TRAJ_NPZ" \
    --out_dir "$VIDEO_OUT_DIR"

# MUJOCO_GL=egl /home/yochin/miniforge3/envs/starVLA/bin/python examples/G1_Replay/eval_files/render_replay_sync.py \
#     --npz results/Checkpoints/starvla_qwengr00t_g1_dexhands_fridge_pick_grapes/pred_traj_full.npz \
#     --out_dir results/Checkpoints/starvla_qwengr00t_g1_dexhands_fridge_pick_grapes/replay_videos_sync

echo "=========================================================="
echo "렌더링 완료! 결과 비디오 저장 경로: $VIDEO_OUT_DIR"
echo "=========================================================="
