#!/bin/bash
set -e

# 6D state run 을 80k -> 160k 로 이어서 학습한다. my_run_g1_fridge5_state6d.sh 의
# 후속이며 같은 run_id 안에서 진행되므로 체크포인트와 W&B 가 이어진다.
#
# 왜 이어서 하는가: 스텝을 맞춰 비교해 보니 6D 가 쿼터니언보다 모든 항목에서 나았다
# (81D@75k 대비 val 14%, train 21%, base 회전 20배). 표현 비교는 끝났고 남은 변수는
# 스텝이다. 81D run 의 val 은 75k 에서 200k 까지 22% 더 좋아졌으므로(0.02374 ->
# 0.01862) 80k 는 수렴 전이다.
#
# 앞서 80k 를 고른 근거였던 체크포인트 추이 측정(val 이 25k 에 포화)은 표본이 너무
# 적어 실제 개선을 못 본 것이었다. 중간 판단은 전체 평가로만 해야 한다.
#
# 주의: resume 은 가중치만 복원하고 Adam 모멘트는 0 에서 다시 시작하며, LR 은
# 160k 코사인의 80k 지점(피크의 51%)으로 뛴다. 자세한 내용은 resume yaml 의
# trainer 주석에 있다.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# export: the resume-target guard below reads it from a python heredoc.
export REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHON="/home/yochin/miniforge3/envs/starVLA/bin/python"
export ACCELERATE="/home/yochin/miniforge3/envs/starVLA/bin/accelerate"
export MUJOCO_GL=egl

CONFIG_YAML="examples/simBenchmarks/Robocasa_365/train_files/starvla_qwenoft_g1_fridge5_state6d_resume160k.yaml"
RUN_ID="starvla_qwenoft_g1_fridge5_state6d"
RUN_DIR="$REPO_ROOT/results/Checkpoints/$RUN_ID"

# resume 대상이 실제로 80k 인지 먼저 확인한다. _get_latest_checkpoint 는
# checkpoints/ 에서 가장 큰 steps_<N> 을 고르므로, 80k 파일이 없으면 60k 에서
# 조용히 재개되어 20k 를 버린다. 그 사실은 로그를 뒤져야 드러나므로 여기서 막는다.
echo ">> resume 대상 확인"
LATEST=$($PYTHON - <<'PY'
import os, re
d = os.path.join(os.environ["REPO_ROOT"], "results/Checkpoints",
                 "starvla_qwenoft_g1_fridge5_state6d", "checkpoints")
c = [(int(m.group(1)), f) for f in os.listdir(d)
     if (m := re.match(r"steps_(\d+)_pytorch_model\.pt$", f))]
print(max(c)[0] if c else 0)
PY
)
echo "   checkpoints/ 최신 스텝: $LATEST"
if [ "$LATEST" != "80000" ]; then
    echo "!! 80000 이어야 합니다. final_model/pytorch_model.pt 를"
    echo "   checkpoints/steps_80000_pytorch_model.pt 로 복사했는지 확인하세요."
    exit 1
fi

# 80k 평가 산출물이 보존되어 있는지 확인한다. 아래 평가는 eval_160k/ 로 쓰지만,
# 80k 결과가 없으면 비교 대상이 사라진 것이므로 진행할 이유가 없다.
if [ ! -f "$RUN_DIR/eval_80k/openloop_mse.json" ]; then
    echo "!! $RUN_DIR/eval_80k/openloop_mse.json 이 없습니다."
    echo "   80k 평가 결과를 eval_80k/ 로 보존한 뒤 다시 실행하세요."
    exit 1
fi
echo "   80k 평가 결과 보존 확인됨"

echo "=========================================================="
echo " G1 Fridge 5태스크 QwenOFT 6D — 80k -> 160k 이어 학습"
echo "  설정: $CONFIG_YAML"
echo "  재개 LR: base 5.3e-6 / action_model 5.1e-5 (피크의 51%, 워밍업 없음)"
echo "  Adam 모멘트는 복원되지 않으므로 초반 loss 스파이크는 정상"
echo "  저장: 100k / 120k / 140k / 160k"
echo "=========================================================="

WANDB_MODE=online \
TOKENIZERS_PARALLELISM=false \
$ACCELERATE launch \
    --config_file "$REPO_ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
    --num_processes 1 \
    "$REPO_ROOT/starVLA/training/train_starvla.py" \
    --config_yaml "$REPO_ROOT/$CONFIG_YAML"

echo "=========================================================="
echo " 학습 완료. 평가 시작 (결과는 eval_160k/ 로)"
echo "=========================================================="

CKPT="$RUN_DIR/final_model/pytorch_model.pt"
E="$REPO_ROOT/examples/G1_Replay/eval_files"
OUT="$RUN_DIR/eval_160k"
mkdir -p "$OUT"

export G1_EVAL_DATASETS="FridgeApple,FridgeGraspLast,FridgeOnion,FridgePickCoke,FridgeTakeCoke"
EVAL_OPTS="--split both --target state --state_lead 2 --no_send_state"
echo ">> 평가 대상: $G1_EVAL_DATASETS"

echo ">> 1/4 오픈루프 MSE"
$PYTHON "$E/openloop_mse.py" --trained_ckpt "$CKPT" $EVAL_OPTS --out "$OUT/openloop_mse.json"

echo ">> 2/4 관절별 오차 (83D 평가 후 명령 45D 로 저장)"
$PYTHON "$E/perjoint_mse.py" --ckpt "$CKPT" $EVAL_OPTS --out "$OUT/perjoint.json"

echo ">> 3/4 손끝 위치 오차 (45D npz 사용)"
$PYTHON "$E/ee_error_attrib.py" --npz "$OUT/perjoint.npz" --hand right \
    --out "$OUT/ee_attrib_right.json"

# 렌더링은 MuJoCo/EGL 쪽에서 산발적으로 abort 한다. 80k 때도 10개 중 8개까지
# 만들고 죽었다(코어 덤프, 파이썬 예외 아님). 선택적 산출물이므로 파이프라인을
# 멈추지 않는다.
echo ">> 4/4 궤적 덤프 및 MuJoCo 영상 (실패해도 계속)"
{
    $PYTHON "$E/dump_pred_traj.py" --ckpt "$CKPT" $EVAL_OPTS \
        --num_chunks 60 --episodes_per_task 1 --out "$OUT/pred_traj_full.npz" &&
    $PYTHON "$E/render_replay_sync.py" --npz "$OUT/pred_traj_full.npz" \
        --out_dir "$OUT/replay_videos_sync"
} || echo "   [경고] 덤프/렌더링 실패 — 수치 결과는 위에 보존됨"

echo "=========================================================="
echo " ✅ 160k 학습 및 평가 완료"
echo "   80k 결과:  $RUN_DIR/eval_80k/"
echo "   160k 결과: $OUT/"
echo "=========================================================="
