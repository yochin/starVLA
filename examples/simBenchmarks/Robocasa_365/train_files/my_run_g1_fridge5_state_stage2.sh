#!/bin/bash
set -e

# G1 냉장고 5태스크, state 타깃 2차 학습 (1차 200k 에서 이어서 100k).
#
# 1차 학습은 마지막 20k 구간에서도 loss 가 5.9% 내려가 200k 에서 수렴했다고 보기
# 어려웠다. 그래서 절반 길이로 더 돌린다.
#
# 주의: 체크포인트에는 가중치만 있고 옵티마이저 상태가 없다. 따라서 Adam 모멘트는
# 초기화되고 코사인 스케줄도 1e-4 에서 다시 시작하는 warm restart 가 된다.
# 재시작 직후 수백 스텝 동안 loss 가 튀는 것은 이 때문이며, 곧 회복해야 정상이다.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHON="/home/yochin/miniforge3/envs/starVLA/bin/python"
export ACCELERATE="/home/yochin/miniforge3/envs/starVLA/bin/accelerate"
export MUJOCO_GL=egl

CONFIG_YAML="examples/simBenchmarks/Robocasa_365/train_files/starvla_qwenoft_g1_fridge5_state_stage2.yaml"
RUN_ID="starvla_qwenoft_g1_fridge5_state_stage2"
RUN_DIR="$REPO_ROOT/results/Checkpoints/$RUN_ID"
BASE_CKPT="$REPO_ROOT/results/Checkpoints/starvla_qwenoft_g1_fridge5_state/final_model/pytorch_model.pt"

# 이어받을 체크포인트가 실제로 있는지 먼저 확인한다. 없으면 설정만 보고는
# 처음부터 학습하는 것과 구분되지 않아 100k 를 통째로 낭비하게 된다.
if [ ! -f "$BASE_CKPT" ]; then
    echo "!! 1차 학습의 final_model 이 없습니다: $BASE_CKPT"
    exit 1
fi
echo ">> 이어받을 체크포인트: $(du -h "$BASE_CKPT" | cut -f1)  $BASE_CKPT"

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
            echo "   [불완전] $d/$split  에피소드 $n_ep / parquet $n_pq / 비디오 $n_vid"
            MISSING=1
        else
            echo "   [정상]   $d/$split  에피소드 $n_ep"
        fi
    done
done
if [ "$MISSING" -ne 0 ]; then
    echo "!! 데이터가 불완전합니다."
    exit 1
fi

echo "=========================================================="
echo " G1 Fridge 5태스크 state 타깃 2차 학습 (100,000 스텝)"
echo "  설정: $CONFIG_YAML"
echo "  기반: 1차 200k final_model (warm restart — Adam 상태는 초기화됨)"
echo "  Loss 제외 34차원: 속도 29 (base 각속도 3 + 관절 속도 26) + 손 이산 모드 2"
echo "  쿼터니언 전용 loss: state dims 3~6"
echo "  실제 학습 대상: 관절위치 43차원 + 쿼터니언 4차원 = 47차원"
echo "=========================================================="

WANDB_MODE=online \
TOKENIZERS_PARALLELISM=false \
$ACCELERATE launch \
    --config_file "$REPO_ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
    --num_processes 1 \
    "$REPO_ROOT/starVLA/training/train_starvla.py" \
    --config_yaml "$REPO_ROOT/$CONFIG_YAML"

echo "=========================================================="
echo " 학습 완료. 평가 시작 (state 타깃)"
echo "=========================================================="

CKPT="$RUN_DIR/final_model/pytorch_model.pt"
E="$REPO_ROOT/examples/G1_Replay/eval_files"

# state 타깃 체크포인트이므로 --target state 가 필수다. 빠뜨리면 45D 액션 GT 로
# 81D 예측을 재려다 실패한다. state_lead 는 학습 설정의 state_lead_frames 와 맞춘다.
export G1_EVAL_DATASETS="FridgeApple,FridgeGraspLast,FridgeOnion,FridgePickCoke,FridgeTakeCoke"
EVAL_OPTS="--split both --target state --state_lead 2 --no_send_state"
echo ">> 평가 대상: $G1_EVAL_DATASETS"

echo ">> 1/4 오픈루프 MSE"
$PYTHON "$E/openloop_mse.py" --trained_ckpt "$CKPT" $EVAL_OPTS --out "$RUN_DIR/openloop_mse.json"

echo ">> 2/4 관절별 오차 (81D 평가 후 명령 45D 로 저장)"
$PYTHON "$E/perjoint_mse.py" --ckpt "$CKPT" $EVAL_OPTS --out "$RUN_DIR/perjoint.json"

echo ">> 3/4 손끝 위치 오차 (45D npz 사용)"
$PYTHON "$E/ee_error_attrib.py" --npz "$RUN_DIR/perjoint.npz" --hand right \
    --out "$RUN_DIR/ee_attrib_right.json"

# 렌더링은 MuJoCo/EGL 쪽에서 한 번 abort 한 적이 있다. 선택적 산출물이므로
# 실패해도 위 수치 결과가 남도록 여기서 파이프라인을 멈추지 않는다.
echo ">> 4/4 궤적 덤프 및 MuJoCo 영상 (실패해도 계속)"
{
    $PYTHON "$E/dump_pred_traj.py" --ckpt "$CKPT" $EVAL_OPTS \
        --num_chunks 60 --episodes_per_task 1 --out "$RUN_DIR/pred_traj_full.npz" &&
    $PYTHON "$E/render_replay_sync.py" --npz "$RUN_DIR/pred_traj_full.npz" \
        --out_dir "$RUN_DIR/replay_videos_sync"
} || echo "   [경고] 덤프/렌더링 실패 — 수치 결과는 위에 보존됨"

echo "=========================================================="
echo " ✅ 2차 학습 및 평가 완료. 결과: $RUN_DIR"
echo "=========================================================="
