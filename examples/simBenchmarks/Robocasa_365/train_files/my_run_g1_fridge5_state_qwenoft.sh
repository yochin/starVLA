#!/bin/bash
set -e

# G1 냉장고 5개 태스크, QwenOFT 가 action 대신 "0.08초 뒤 state 81차원" 을 예측하도록 학습.
#
# action 은 SONIC 디코더의 명령값이고 state 는 실제 관절 인코더 측정값이라 같은 신호가
# 아니다. 이 데이터에서 실측한 결과 팔은 인코더가 명령을 2~4 프레임 뒤따르고, 다리는
# 상관 0.25~0.55 로 느슨하게만 따라가며(밸런스 제어기 개입), 허리 roll/pitch 는 거의
# 움직이지 않는다. 즉 action 은 로봇이 실제로 한 동작을 설명하지 못한다. state 는 그것을
# 설명하고, 추가로 action 에 채널 자체가 없는 정보(base 자세·각속도, 관절 속도)를 담는다.
#
# 타깃: state[t+2 .. t+17] 81차원. 2프레임 리드는 30 FPS 에서 0.067초로 요청하신 0.08초에
# 가장 가깝고, 값은 UnitreeG1DexHandsStateTargetDataConfig.state_lead_frames 에 있다.
#
# 손 이산 모드(state 기준 24·69)는 L1 대상이 아니라 제외하고, base 쿼터니언(3:7)은
# 1 - |<q_pred, q_target>| 로 따로 채점한다.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHON="/home/yochin/miniforge3/envs/starVLA/bin/python"
export ACCELERATE="/home/yochin/miniforge3/envs/starVLA/bin/accelerate"
export MUJOCO_GL=egl

CONFIG_YAML="examples/simBenchmarks/Robocasa_365/train_files/starvla_qwenoft_g1_fridge5_state.yaml"
RUN_ID="starvla_qwenoft_g1_fridge5_state"
RUN_DIR="$REPO_ROOT/results/Checkpoints/$RUN_ID"

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
    echo "!! 데이터 복사가 끝나지 않았습니다."
    exit 1
fi

echo "=========================================================="
echo " G1 Fridge 5태스크 QwenOFT 학습 (타깃: 0.08초 뒤 state 81D)"
echo "  설정: $CONFIG_YAML"
echo "  Loss 제외 34차원: 속도 29 (base 각속도 3 + 관절 속도 26) + 손 이산 모드 2"
echo "  쿼터니언 전용 loss: state dims 3~6"
echo "  실제 학습 대상: 관절위치 43차원 + 쿼터니언 4차원 = 47차원"
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
echo " 학습 완료. 결과: $RUN_DIR"
echo "=========================================================="
echo "주의: 기존 평가 파이프라인(openloop_mse.py 등)은 GT 를 action 에서 읽고"
echo "45차원을 가정하므로 이 체크포인트에는 그대로 쓸 수 없습니다."
echo "state 기준 GT 와 81차원을 다루도록 평가 코드를 먼저 수정해야 합니다."
