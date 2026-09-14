"""데이터팩토리 G1 변환본의 modality.json 교정 (파케이 재작성 없음).

납품 데이터는 `observation.state`와 `action` 컬럼이 **같은 126D 벡터로 복제**돼 있다
(변환기가 원본 20개 컬럼을 이름 알파벳순으로 concat 후 양쪽에 복사). 그대로 학습하면
모델이 항등함수를 배운다. 다행히 126D 안에 진짜 액션과 진짜 관측이 모두 들어 있어,
modality.json의 슬라이스 정의만 고치면 된다.

키 이름 주의: starVLA 로더가 `key.replace(modality + ".", "")` 를 **전체 치환**으로
수행해서, 원본 컬럼명(`action.g1.action.left_arm...`)을 쓰면 키가 깨진다. 그래서
'state.'/'action.' 부분문자열이 없는 깨끗한 별칭으로 정의한다.

원본은 modality.json.orig 로 백업한다(멱등 — 재실행해도 백업을 덮어쓰지 않음).

실행:
  python fix_modality.py /path/to/LEROBOT_ROOT/<dataset_dir> [...]
  # 예: python fix_modality.py $G1_DATA_ROOT/fridge_grapes_{train,val}
"""

import json
import shutil
import sys
from pathlib import Path

# observation.state 컬럼(126D) 안의 진짜 관측 슬라이스
STATE = {
    "left_arm_pos":  (54, 61),  "right_arm_pos": (99, 106),
    "legs_pos":      (75, 87),  "waist_pos":     (120, 123),
    "left_arm_vel":  (61, 68),  "right_arm_vel": (106, 113),
    "legs_vel":      (87, 99),  "waist_vel":     (123, 126),
    "base_angvel":   (45, 48),  "base_quat":     (48, 52),
    # 아래 3개는 센서가 아니라 명령 복사본(에코)이다 — 학습 state에 넣지 말 것.
    # 참고·검증용으로만 정의를 남긴다.
    "left_hand_pos_echo": (68, 75), "right_hand_pos_echo": (113, 120),
    "head_pos_echo": (52, 54),
}

# action 컬럼(=동일 126D) 안의 진짜 액션 슬라이스
ACTION = {
    "left_arm": (2, 9), "right_arm": (28, 35), "left_hand": (9, 16),
    "right_hand": (35, 42), "legs": (16, 28), "waist": (42, 45), "head": (0, 2),
}


def fix(ds_dir: Path) -> None:
    p = ds_dir / "meta" / "modality.json"
    if not p.is_file():
        print(f"  [skip] {ds_dir} — meta/modality.json 없음")
        return
    orig = p.with_suffix(".json.orig")
    if not orig.exists():
        shutil.copy2(p, orig)
    m = json.load(open(orig))
    m["state"] = {k: {"original_key": "observation.state", "start": a, "end": b}
                  for k, (a, b) in STATE.items()}
    m["action"] = {k: {"original_key": "action", "start": a, "end": b}
                   for k, (a, b) in ACTION.items()}
    json.dump(m, open(p, "w"), indent=2)
    print(f"  [ok] {ds_dir.name}: action {sum(b-a for a,b in ACTION.values())}D / "
          f"state {sum(b-a for a,b in STATE.values())}D(에코 16D 포함)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        fix(Path(d).resolve())
    print("\n주의: 학습 data_config 의 state_keys 에서 *_echo 3개는 제외할 것 (81D → 65D).")
