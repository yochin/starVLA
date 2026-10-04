#!/bin/bash
set -e

# G1 냉장고 5태스크. 6D state 타깃에 state[t] 를 입력으로 주고, 헤드가 그로부터의
# 잔차를 예측한다. my_run_g1_fridge5_state6d.sh 와 스텝(80k)·스케줄·데이터가 같으므로
# state 없는 6D@80k 와 바로 짝 비교가 된다.
#
# 왜 잔차인가: 160k 6D run 의 오차가 지평에 거의 무관하게 평탄했다(val RMSE 0.1056
# at 0.07s -> 0.1377 at 0.57s). 같은 구간에서 persistence 기준선은 0.0444 -> 0.1401 로
# 세 배가 된다. 평탄한 오차는 "움직임을 못 맞춘다"가 아니라 "현재 절대 관절자세를
# 픽셀에서 못 짚는다"는 고정 바닥의 징후다. state[t] 를 주면 절대값을 추정할 필요 없이
# "여기서 조금" 을 내면 된다.
#
# 주의 1: 평가에서 --no_send_state 를 쓰면 안 된다. 이 모델은 state 입력이 필수다.
# 주의 2: 평가 클라이언트가 보내는 state 는 원시 라디안이고 학습은 정규화 state 를
#   썼다. 그 불일치는 predict_action 의 state_is_normalized=False 를 래퍼가 실제로
#   처리하게 고쳐서 해결했다(이전에는 무시되는 죽은 인자였고, 그 때문에 과거
#   GR00T+state 실험이 전부 잘못 평가됐다).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHON="/home/yochin/miniforge3/envs/starVLA/bin/python"
export ACCELERATE="/home/yochin/miniforge3/envs/starVLA/bin/accelerate"
export MUJOCO_GL=egl

CONFIG_YAML="examples/simBenchmarks/Robocasa_365/train_files/starvla_qwenoft_g1_fridge5_state6dreshand_E_resume160k.yaml"
RUN_ID="starvla_qwenoft_g1_fridge5_state6dreshand_E"
RUN_DIR="$REPO_ROOT/results/Checkpoints/$RUN_ID"

# 6D 경로는 modality.json 의 rotation_type 에 의존한다. 없으면 쿼터니언 4D 가 그대로
# 흘러가 83차원을 기대하는 설정과 어긋난 채 학습이 시작되고, 19시간 뒤에야 드러난다.
echo ">> 데이터셋 및 회전 메타 확인"
MISSING=0
for d in FridgeApple FridgeGraspLast FridgeOnion FridgePickCoke FridgeTakeCoke; do
    for split in train val; do
        p="$REPO_ROOT/playground/Datasets/$d/$split"
        if [ ! -f "$p/meta/info.json" ]; then
            echo "   [없음] $d/$split"; MISSING=1; continue
        fi
        rot=$($PYTHON -c "
import json
m=json.load(open('$p/meta/modality.json'))
print(m['state'].get('g1.observation.base.orientation',{}).get('rotation_type','없음'))")
        if [ "$rot" != "quaternion" ]; then
            echo "   [회전메타 없음] $d/$split rotation_type=$rot"; MISSING=1
        fi
    done
done
[ "$MISSING" -ne 0 ] && { echo "!! 데이터/회전 메타 미비"; exit 1; }
echo "   전 스플릿 rotation_type=quaternion 확인"

# 타깃 누출 가드. 이 설정은 state 모달리티에 현재 프레임(delta 0)을 앞에 붙이고
# 로더가 그 행을 입력으로 떼어낸다. 붙지 않았다면 state 입력이 곧 타깃이 되어
# 모델에 정답을 주게 되는데, loss 가 그냥 낮아질 뿐 오류는 나지 않는다.
echo ">> 타깃 누출 가드"
$PYTHON - <<'PY'
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"],
    "examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files"))
from data_registry.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
cfg = ROBOT_TYPE_CONFIG_MAP["unitree_g1_dexhands_state_target_6d_res_hand"]
di = list(cfg.modality_config()["state"].delta_indices)
assert di[0] == 0 and len(di) == 17, f"state delta_indices 가 [0, lead..] 가 아니다: {di}"
ds = LeRobotSingleDataset(
    dataset_path=os.path.join(os.environ["REPO_ROOT"], "playground/Datasets/FridgeApple/train"),
    modality_configs=cfg.modality_config(), transforms=cfg.transform(),
    embodiment_tag="new_embodiment", video_backend="torchvision_av",
    data_cfg={"action_target": "state", "include_state": True, "state_current_row": True,
              "hand_labels": True})
s = ds[0]
st = np.asarray(s["state"], dtype=np.float64); ac = np.asarray(s["action"], dtype=np.float64)
assert st.shape == (1, 83), st.shape
assert ac.shape == (16, 83), ac.shape
leak = np.abs(ac[0] - st[0]).mean()
assert leak > 1e-4, f"state 입력이 타깃과 동일하다(누출): 차이 {leak}"
print(f"   state {st.shape} / 타깃 {ac.shape} / |타깃[0]-state| = {leak:.5f} (>0, 누출 없음)")
for k, want in [("hand_on", (16, 2)), ("hand_mode", (16, 2)), ("hand_thumb_flex", (16, 2)), ("hand_prev", (2, 3))]:
    a = np.asarray(s[k])
    assert a.shape == want, f"{k} 형태가 {a.shape}, {want} 이어야 한다"
print(f"   손 라벨 확인: hand_on/mode/thumb_flex (16,2), hand_prev (2,3)")
PY

# resume 은 checkpoints/ 의 최대 steps_<N> 에서 재개한다. 80k 미만이면 스텝을
# 버리게 되고, 0 이면 처음부터 학습한다. 둘 다 조용히 일어나므로 막는다.
echo ">> resume 대상 확인"
LATEST=$($PYTHON -c "
import os,re
d=os.path.join('$RUN_DIR','checkpoints')
c=[int(m.group(1)) for f in os.listdir(d) if (m:=re.match(r'steps_(\\d+)_pytorch_model\\.pt$',f))]
print(max(c) if c else 0)")
echo "   checkpoints/ 최신 스텝: $LATEST"
if [ -z "$LATEST" ] || [ "$LATEST" -lt 80000 ] || [ "$LATEST" -ge 160000 ]; then
    echo "!! 재개 지점이 80000 이상 160000 미만이어야 합니다 (현재 $LATEST)"; exit 1
fi
# final_model 이 덮어써지므로 80k 평가 결과가 먼저 보존돼 있어야 한다.
if [ ! -f "$RUN_DIR/eval_80k/openloop_mse.json" ]; then
    echo "!! eval_80k/openloop_mse.json 이 없습니다 — 80k 결과를 먼저 보존하세요"; exit 1
fi
echo "   80k 평가 결과 보존 확인됨"

echo "=========================================================="
echo " G1 Fridge 5태스크 QwenOFT 6D + 손 분류기 (E 설정) — 80k->160k 연장"
echo "  설정: $CONFIG_YAML"
echo "  모델 입력: 이미지 4장 + 언어 (state 는 입력이 아니다)"
echo "  state[t]: body 83차원 출력에 더해지는 잔차 기준점"
echo "  출력: body 83D + 손 로짓 12D(on/모드4/엄지굽힘 x 2) = 95D"
echo "  감독 37차원: 6D회전 6 + head 2 + 팔 14 + 다리 12 + 허리 3"
echo ""
echo "  D 와의 차이: 손 로짓 전용 MLPResNet (트렁크까지 분리)"
echo "  손 run 과의 차이 두 가지:"
echo "   1) hand_prev 프롬프트 제거 — 손 분류기가 현재 자세를 보지 않는다"
echo "   2) head 9,10 을 L1 로 복귀 — 미감독이면 쓰레기값(val 78배)이 나온다"
echo "  손 회귀 차원 25-31,70-76 은 전부 제외: 분류기가 손을 전담한다"
echo "  스텝: 80,000 / 손 pos_weight 1.5 (손 run 과 동일, 변수 고정)"
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
OUT="$RUN_DIR/eval_160k"
mkdir -p "$OUT"
E="$REPO_ROOT/examples/G1_Replay/eval_files"

export G1_EVAL_DATASETS="FridgeApple,FridgeGraspLast,FridgeOnion,FridgePickCoke,FridgeTakeCoke"
# --no_send_state 없음: 이 모델은 state 없이는 예측할 수 없다.
EVAL_OPTS="--split both --target state --state_lead 2"
echo ">> 평가 대상: $G1_EVAL_DATASETS (state 전송)"

echo ">> 1/4 오픈루프 MSE"
$PYTHON "$E/openloop_mse.py" --trained_ckpt "$CKPT" $EVAL_OPTS --out "$OUT/openloop_mse.json"

echo ">> 2/4 관절별 오차"
$PYTHON "$E/perjoint_mse.py" --ckpt "$CKPT" $EVAL_OPTS --out "$OUT/perjoint.json"

echo ">> 3/4 손끝 위치 오차"
$PYTHON "$E/ee_error_attrib.py" --npz "$OUT/perjoint.npz" --hand right \
    --out "$OUT/ee_attrib_right.json"

# 렌더링은 MuJoCo/EGL 에서 산발적으로 네이티브 abort 한다(6D run 에서도 10개 중
# 8개까지 만들고 코어 덤프). 선택적 산출물이라 파이프라인을 멈추지 않는다.
echo ">> 4/4 궤적 덤프 및 영상 (실패해도 계속)"
{
    $PYTHON "$E/dump_pred_traj.py" --ckpt "$CKPT" $EVAL_OPTS \
        --num_chunks 60 --episodes_per_task 1 --out "$OUT/pred_traj_full.npz" &&
    $PYTHON "$E/render_replay_sync.py" --npz "$OUT/pred_traj_full.npz" \
        --out_dir "$OUT/replay_videos_sync"
} || echo "   [경고] 덤프/렌더링 실패 — 수치 결과는 보존됨"

echo "=========================================================="
echo " ✅ 완료. 결과: $RUN_DIR"
echo "  비교 대상: results/Checkpoints/starvla_qwenoft_g1_fridge5_state6d/eval_80k/"
echo "=========================================================="
