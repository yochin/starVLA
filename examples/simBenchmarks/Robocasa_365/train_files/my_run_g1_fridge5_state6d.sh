#!/bin/bash
set -e

# G1 냉장고 5개 태스크. QwenOFT 가 action 대신 "0.08초 뒤 state 83차원" 을 예측한다.
#
# 왜 action 이 아니라 state 인가: action 은 SONIC 디코더의 명령값이고 state 는 실제 관절
# 인코더 측정값이라 같은 신호가 아니다. 실측 결과 팔은 인코더가 명령을 2~4 프레임 뒤따르고,
# 다리는 상관 0.25~0.55 로 느슨하게만 따라가며(밸런스 제어기 개입), 허리 roll/pitch 는 거의
# 움직이지 않는다. 즉 action 은 로봇이 실제로 한 동작을 설명하지 못한다.
#
# 왜 83차원인가: base 자세를 쿼터니언 4D 가 아니라 6D 회전으로 싣기 때문이다. 쿼터니언을
# 그대로 회귀하면 q 와 -q 가 같은 회전이라 표현이 불연속이고, 81D run 에서 base_quat 의
# val 오차가 persistence 기준선의 328배였다. 6D 는 회전행렬의 첫 두 행이고 세 번째 행은
# 외적으로 복원되므로 정보 손실 없이 연속이다. 커스텀 loss 가 필요 없어져 49개 감독
# 차원 전부가 같은 L1 으로 학습된다.
#
# 타깃 시점: state[t+2 .. t+17]. 2프레임 리드는 30 FPS 에서 0.067초이고, 값은
# UnitreeG1DexHandsStateTargetDataConfig.state_lead_frames 에 있다.
#
# 스텝 수 80k 의 근거: 81D run 의 체크포인트 추이를 재보니 val 이 25k 에서 이미 포화했다
# (val 0.03000 -> 0.02962 -> 0.02980 at 25k/100k/200k). train 만 3배 좋아지고 val 은
# 멈춰 있었으므로 200k 는 낭비다. 다만 25k~100k 구간의 차이는 노이즈 수준이라 80k 는
# "평탄 구간 안에서 6D 학습 여유를 조금 둔" 선택이다.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHON="/home/yochin/miniforge3/envs/starVLA/bin/python"
export ACCELERATE="/home/yochin/miniforge3/envs/starVLA/bin/accelerate"
export MUJOCO_GL=egl

CONFIG_YAML="examples/simBenchmarks/Robocasa_365/train_files/starvla_qwenoft_g1_fridge5_state6d.yaml"
RUN_ID="starvla_qwenoft_g1_fridge5_state6d"
RUN_DIR="$REPO_ROOT/results/Checkpoints/$RUN_ID"

# 6D 경로는 modality.json 의 rotation_type 에 의존한다. 없으면 RotationTransform 이
# 초기화되지 않아 쿼터니언 4D 가 그대로 흘러가고, 83차원을 기대하는 설정과 어긋난 채
# 학습이 시작된다. 그 어긋남은 19시간 뒤에야 드러나므로 여기서 먼저 막는다.
echo ">> 데이터셋 및 회전 메타 확인"
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
        rot=$($PYTHON -c "
import json
m=json.load(open('$p/meta/modality.json'))
print(m['state'].get('g1.observation.base.orientation',{}).get('rotation_type','없음'))")
        if [ "$n_ep" -ne "$n_pq" ] || [ "$n_vid" -ne $((n_ep * 3)) ]; then
            echo "   [불완전] $d/$split  에피소드 $n_ep / parquet $n_pq / 비디오 $n_vid"
            MISSING=1
        elif [ "$rot" != "quaternion" ]; then
            echo "   [회전메타 없음] $d/$split  rotation_type=$rot  (quaternion 이어야 함)"
            MISSING=1
        else
            echo "   [정상]   $d/$split  에피소드 $n_ep  rotation_type=$rot"
        fi
    done
done
if [ "$MISSING" -ne 0 ]; then
    echo "!! 데이터 또는 회전 메타가 준비되지 않았습니다."
    exit 1
fi

echo "=========================================================="
echo " G1 Fridge 5태스크 QwenOFT — 타깃: 0.08초 뒤 state 83D"
echo "  설정: $CONFIG_YAML"
echo "  base 회전: 6D 표현 [3:9] — 커스텀 loss 없이 L1 로 학습"
echo "  Loss 제외 34차원: 속도 29 (base 각속도 3 + 관절 속도 26) + 손 이산 모드 (26, 71)"
echo "  실제 학습 대상: 관절위치 43 + 6D 회전 6 = 49차원"
echo "  스텝: 80,000 (코사인 스케줄러, 저장 20k 마다)"
echo "  State 입력: 미사용 (include_state=false — 켜면 복사 과제가 됨)"
echo "=========================================================="

WANDB_MODE=online \
TOKENIZERS_PARALLELISM=false \
$ACCELERATE launch \
    --config_file "$REPO_ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
    --num_processes 1 \
    "$REPO_ROOT/starVLA/training/train_starvla.py" \
    --config_yaml "$REPO_ROOT/$CONFIG_YAML"

echo "=========================================================="
echo " 학습 완료. 평가 시작"
echo "=========================================================="

CKPT="$RUN_DIR/final_model/pytorch_model.pt"
E="$REPO_ROOT/examples/G1_Replay/eval_files"

# --target state 가 없으면 45D 액션 GT 로 83D 예측을 재려다 실패한다.
# state_lead 는 학습 설정의 state_lead_frames 와 맞춘다.
export G1_EVAL_DATASETS="FridgeApple,FridgeGraspLast,FridgeOnion,FridgePickCoke,FridgeTakeCoke"
EVAL_OPTS="--split both --target state --state_lead 2 --no_send_state"
echo ">> 평가 대상: $G1_EVAL_DATASETS"

echo ">> 1/4 오픈루프 MSE"
$PYTHON "$E/openloop_mse.py" --trained_ckpt "$CKPT" $EVAL_OPTS --out "$RUN_DIR/openloop_mse.json"

echo ">> 2/4 관절별 오차 (83D 평가 후 명령 45D 로 저장)"
$PYTHON "$E/perjoint_mse.py" --ckpt "$CKPT" $EVAL_OPTS --out "$RUN_DIR/perjoint.json"

echo ">> 3/4 손끝 위치 오차 (45D npz 사용)"
$PYTHON "$E/ee_error_attrib.py" --npz "$RUN_DIR/perjoint.npz" --hand right \
    --out "$RUN_DIR/ee_attrib_right.json"

# 렌더링은 MuJoCo/EGL 쪽에서 산발적으로 abort 한 적이 있다. 선택적 산출물이므로
# 실패해도 위 수치 결과가 남도록 파이프라인을 멈추지 않는다.
echo ">> 4/4 궤적 덤프 및 MuJoCo 영상 (실패해도 계속)"
{
    $PYTHON "$E/dump_pred_traj.py" --ckpt "$CKPT" $EVAL_OPTS \
        --num_chunks 60 --episodes_per_task 1 --out "$RUN_DIR/pred_traj_full.npz" &&
    $PYTHON "$E/render_replay_sync.py" --npz "$RUN_DIR/pred_traj_full.npz" \
        --out_dir "$RUN_DIR/replay_videos_sync"
} || echo "   [경고] 덤프/렌더링 실패 — 수치 결과는 위에 보존됨"

echo "=========================================================="
echo " ✅ 6D 학습 및 평가 완료. 결과: $RUN_DIR"
echo "=========================================================="
