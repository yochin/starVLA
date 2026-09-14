#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHON="/home/yochin/miniforge3/envs/starVLA/bin/python"

export CKPT="$REPO_ROOT/results/Checkpoints/starvla_qwengr00t_g1_dexhands_fridge_pick_grapes/final_model/pytorch_model.pt"
export RUN_DIR="$REPO_ROOT/results/Checkpoints/starvla_qwengr00t_g1_dexhands_fridge_pick_grapes"

echo "=== 1. 궤적 플롯 및 Horizon Drift 그래프 생성 ==="
$PYTHON "$SCRIPT_DIR/eval_files/plot_openloop_traj.py" \
    --ckpt "$CKPT" \
    --split both \
    --num_chunks 60 \
    --out_dir "$RUN_DIR/traj_plots"

echo "=== 2. Step 0~15별 수치 분석 및 Train vs Val MSE 산출 ==="
$PYTHON "$SCRIPT_DIR/eval_files/openloop_mse.py" \
    --trained_ckpt "$CKPT" \
    --split both \
    --out "$RUN_DIR/openloop_mse.json"