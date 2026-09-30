# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""G1 손 명령 코드북 — 학습·평가·배포가 공유한다.

손은 회귀가 아니라 이산 명령이다. 모드가 손 모양을 정하고, on/off 가 그 모양을
취할지 전부 펼지를 정하며, 엄지 굽힘이 별도 축이다. 실제 동작은 세 가지뿐이다:
전부 펴기 / 전부 구부리기 / 엄지 펴고 나머지 구부리기.

여기 값은 5개 데이터셋 141,139 프레임에서 관측된 원시 라디안 명령이다. 정규화
공간이 아니라 원시 값을 쓰는 이유는 좌우 손이 q99 정규화에서 반대 방향으로
매핑되기 때문이다(원시 0 이 왼손에서는 약 +1, 오른손에서는 약 -1 로 간다).
"""

from __future__ import annotations

import numpy as np


# 손 명령 코드북 — 원시 라디안. 모드가 손 모양을 정하고 on/off 가 그 모양을 취할지
# 전부 펼지를 정하며, 엄지 굽힘은 별도 축이다. 값은 5개 데이터셋 141,139 프레임에서
# 관측된 고유 명령 벡터이고 상위 조합이 99.84%(L) / 99.89%(R) 를 덮는다.
#
# 각 항목은 (엄지회전, 검지, 중지, 약지, 소지). 엄지 굽힘은 별도로 0/1 을 넣고,
# off 는 여섯 축이 전부 0 이다. 모드 2 는 이 데이터에서 한 번도 구동되지 않아
# on 항목이 없다(선택되더라도 off 로 나온다).
HAND_ON_CODEBOOK = {
    "L_hand": {
        0: (1.75, -1.57, -1.75, -1.57, -1.75),
        1: (0.00, -1.57, -1.75, -1.57, -1.75),
        3: (1.05, -0.94, 0.00, 0.00, 0.00),
    },
    "R_hand": {
        0: (-1.75, 1.57, 1.75, 1.57, 1.75),
        3: (-1.05, 0.94, 0.00, 0.00, 0.00),
    },
}

# on 항목이 없는 모드를 예측했을 때 대신 쓸 모드. 데이터에 없는 조합을 내보내는
# 것보다 그 손에서 가장 흔한 파악으로 떨어뜨리는 편이 안전하다.
HAND_FALLBACK_MODE = {"L_hand": 1, "R_hand": 0}


def hand_command_vector(hand: str, on: bool, mode: int, thumb_flex: bool) -> np.ndarray:
    """(on, 모드, 엄지굽힘) → 손 7축 원시 명령.

    반환 순서는 레이아웃과 같다: 엄지굽힘, 모드, 엄지회전, 손가락 4축.
    """
    book = HAND_ON_CODEBOOK[hand]
    m = int(mode)
    if not on:
        return np.array([0.0, float(m), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if m not in book:
        m = HAND_FALLBACK_MODE[hand]
    rot, *fing = book[m]
    return np.array(
        [1.0 if thumb_flex else 0.0, float(m), rot, *fing], dtype=np.float32
    )


def fill_hand_from_logits(
    out: np.ndarray, logits: np.ndarray, layout: dict, hands: tuple = ("L_hand", "R_hand")
) -> np.ndarray:
    """손 분류 로짓을 원시 명령값으로 바꿔 out 의 손 구간에 써 넣는다.

    Args:
        out: ``(T, D)`` 비정규화 출력. 제자리에서 수정되지 않고 사본이 반환된다.
        logits: ``(T, n_hands * 6)`` — 손마다 on(1) + mode(4) + thumb_flex(1).
        layout: HAND81 또는 HAND45.
        hands: 로짓 순서와 같은 손 이름.

    Returns:
        손 구간이 채워진 ``out`` 사본.
    """
    out = np.array(out, dtype=np.float32, copy=True)
    T = out.shape[0]
    lg = np.asarray(logits, dtype=np.float32).reshape(T, len(hands), 6)
    for hi, hand in enumerate(hands):
        idx = layout[hand]
        cols = [idx["thumb_flex"], idx["mode"], idx["thumb_rot"], *idx["fingers"]]
        on = lg[:, hi, 0] > 0.0
        mode = lg[:, hi, 1:5].argmax(axis=-1)
        flex = lg[:, hi, 5] > 0.0
        for t in range(T):
            out[t, cols] = hand_command_vector(hand, bool(on[t]), int(mode[t]), bool(flex[t]))
    return out




# ---------------------------------------------------------------------------
# 손 채널 레이아웃과 판정 임계값 (학습·평가·배포 공용)
# ---------------------------------------------------------------------------
HAND81 = {
    "L_hand": {"thumb_flex": 23, "mode": 24, "thumb_rot": 25, "fingers": [26, 27, 28, 29]},
    "R_hand": {"thumb_flex": 68, "mode": 69, "thumb_rot": 70, "fingers": [71, 72, 73, 74]},
}

# 손가락이 "구부러졌다"고 볼 임계값(라디안). 값은 0 또는 ±1.57/±1.75 두 극단에
# 몰려 있고 중간대가 0~3.6% 뿐이므로 0.3 은 넉넉한 경계다. 좌우 부호가 반대라
# 절대값으로 판정한다.
HAND45 = {
    "L_hand": {"thumb_flex": 9, "mode": 10, "thumb_rot": 11, "fingers": [12, 13, 14, 15]},
    "R_hand": {"thumb_flex": 35, "mode": 36, "thumb_rot": 37, "fingers": [38, 39, 40, 41]},
}
HAND_CURL_THRESHOLD = 0.3
HAND_THUMB_FLEX_THRESHOLD = 0.5
