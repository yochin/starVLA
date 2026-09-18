"""81D state 레이아웃 정의와 45D 액션 레이아웃으로의 변환.

state 를 타깃으로 학습한 체크포인트(datasets.vla_data.action_target: state)는
45D 액션이 아니라 81D state 를 내놓는다. 기존 평가 스크립트들은 전부 45D 를
가정하므로, 여기서 한 번 변환해 주면 하위 스크립트(perjoint_mse, ee_error_attrib,
render_replay_sync, dump_pred_traj)는 고치지 않고 그대로 쓸 수 있다.

이 모듈은 기존 코드를 전혀 건드리지 않는 순수 추가분이다. import 하지 않는 한
어떤 동작도 바뀌지 않는다.

레이아웃 출처: data_registry/data_config.py 의 UnitreeG1DexHandsDirectGR00TDataConfig
state_keys 순서와 state_key_dims. 값은 그 정의에서 계산해 검증했다.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 81D state 레이아웃 (state_keys 순서대로 누적)
# ---------------------------------------------------------------------------
STATE81_PARTS: dict[str, tuple[int, int]] = {
    "base_angvel": (0, 3),
    "base_quat": (3, 7),
    "head": (7, 9),
    "L_arm": (9, 16),
    "L_arm_vel": (16, 23),
    "L_hand": (23, 30),
    "legs": (30, 42),
    "legs_vel": (42, 54),
    "R_arm": (54, 61),
    "R_arm_vel": (61, 68),
    "R_hand": (68, 75),
    "waist": (75, 78),
    "waist_vel": (78, 81),
}

STATE81_DIM = 81

# 쿼터니언 구간: L1 이 아니라 회전 전용 loss 로 학습된 구간.
STATE_QUAT_SLICE = STATE81_PARTS["base_quat"]

# 손 이산 모드 셀렉터({0,1,2,3}). 45D 액션에서는 10 / 36 에 해당한다.
STATE_MODE_DIMS = [
    STATE81_PARTS["L_hand"][0] + 1,  # 24
    STATE81_PARTS["R_hand"][0] + 1,  # 69
]

# loss 에서 제외된 차원: 속도 전부 + base 각속도 + 손 이산 모드.
# 예측은 나오지만 학습 신호가 없으므로 지표에서도 빼야 한다.
_UNSUPERVISED_PARTS = ["base_angvel", "L_arm_vel", "legs_vel", "R_arm_vel", "waist_vel"]
STATE_UNSUPERVISED_DIMS = sorted(
    {d for p in _UNSUPERVISED_PARTS for d in range(*STATE81_PARTS[p])}
    | set(STATE_MODE_DIMS)
)

# 실제로 감독되는 차원(관절위치 43 + 쿼터니언 4 = 47).
STATE_SUPERVISED_DIMS = [d for d in range(STATE81_DIM) if d not in set(STATE_UNSUPERVISED_DIMS)]

# 연속값 지표에 쓸 차원: 감독 차원에서 쿼터니언을 뺀 관절위치 43개.
# 쿼터니언은 단위가 달라 관절각 MSE 와 같이 평균내면 의미가 없다.
STATE_CONT_DIMS = [
    d for d in STATE_SUPERVISED_DIMS if not (STATE_QUAT_SLICE[0] <= d < STATE_QUAT_SLICE[1])
]

# ---------------------------------------------------------------------------
# 81D state -> 45D 액션 레이아웃
# ---------------------------------------------------------------------------
# 45D 액션 순서: head2, L_arm7, L_hand7, legs12, R_arm7, R_hand7, waist3
_ACTION45_ORDER = ["head", "L_arm", "L_hand", "legs", "R_arm", "R_hand", "waist"]

ACTION45_FROM_STATE81 = np.array(
    [d for part in _ACTION45_ORDER for d in range(*STATE81_PARTS[part])], dtype=np.int64
)

ACTION45_DIM = 45


def state81_to_action45(arr: np.ndarray) -> np.ndarray:
    """81D state 벡터(마지막 축)에서 명령 가능한 45D 관절위치를 뽑는다.

    Args:
        arr: 마지막 축이 81 인 배열. 앞쪽 축 수는 상관없다.

    Returns:
        마지막 축이 45 인 배열. 45D 액션 레이아웃과 순서가 같으므로
        기존 평가 스크립트에 그대로 넣을 수 있다.
    """
    arr = np.asarray(arr)
    if arr.shape[-1] != STATE81_DIM:
        raise ValueError(f"마지막 축이 {STATE81_DIM} 이어야 하는데 {arr.shape} 이다")
    return arr[..., ACTION45_FROM_STATE81]


if __name__ == "__main__":
    # 레이아웃 자체 검증: 구간이 빈틈없이 81 을 덮고, 45D 추출이 정확한지.
    covered = sorted(d for a, b in STATE81_PARTS.values() for d in range(a, b))
    assert covered == list(range(STATE81_DIM)), "구간이 81D 를 정확히 덮지 않는다"
    assert len(ACTION45_FROM_STATE81) == ACTION45_DIM, len(ACTION45_FROM_STATE81)
    assert STATE_MODE_DIMS == [24, 69], STATE_MODE_DIMS
    assert len(STATE_UNSUPERVISED_DIMS) == 34, len(STATE_UNSUPERVISED_DIMS)
    assert len(STATE_SUPERVISED_DIMS) == 47, len(STATE_SUPERVISED_DIMS)
    assert len(STATE_CONT_DIMS) == 43, len(STATE_CONT_DIMS)

    probe = np.arange(STATE81_DIM, dtype=np.float32)[None, None, :]
    out = state81_to_action45(probe)
    assert out.shape == (1, 1, 45), out.shape
    assert out[0, 0, 0] == 7 and out[0, 0, 1] == 8, out[0, 0, :2]      # head
    assert out[0, 0, 10] == 24, out[0, 0, 10]                           # L_hand 모드
    assert out[0, 0, 36] == 69, out[0, 0, 36]                           # R_hand 모드
    assert out[0, 0, 42] == 75 and out[0, 0, 44] == 77, out[0, 0, 42:]  # waist

    print("레이아웃 검증 통과")
    print(f"  감독 {len(STATE_SUPERVISED_DIMS)}차원 = 관절위치 {len(STATE_CONT_DIMS)} + 쿼터니언 4")
    print(f"  무감독 {len(STATE_UNSUPERVISED_DIMS)}차원 (속도 29 + base 각속도 3 + 손 모드 2)")
    print(f"  45D 추출 인덱스: {ACTION45_FROM_STATE81.tolist()}")
