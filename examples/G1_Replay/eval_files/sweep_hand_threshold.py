"""손 on/off 임계값을 모델 재실행 없이 스윕한다.

openloop_mse.py --dump_hand_logits 이 남긴 npz 를 읽어, on/off 로짓 임계값을
바꿔가며 전이 검출의 정밀도·재현율을 다시 센다. 모델은 한 번만 돌리면 되고
임계값 선택은 전부 여기서 끝난다.

왜 임계값을 따로 고르나: 학습은 임계값 0 을 가정하지만 그 지점이 배치에 좋은
동작점이라는 보장이 없다. 그리고 이 문제에서 두 오류의 값이 다르다 — 비전이
프레임에서 잘못 쥐면 그 프레임 오차가 0.0007 에서 약 2.0 으로 뛰고, 전이를
놓치면 persistence 와 같아질 뿐이다. 그래서 기본 기준은 F1 이 아니라 "정밀도
하한을 만족하는 것 중 재현율 최대"다.

    python sweep_hand_threshold.py --npz <path>.trained_80000.npz

--min_precision 으로 하한을 바꾸고, --split 으로 고르는 기준 스플릿을 바꾼다.
임계값은 val 에서 고르고 train 수치는 참고로만 본다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 로짓 배치: 손마다 [on, 모드 4, 엄지굽힘] 6개. 손 순서는 npz 의 hands 와 같다.
HAND_BLOCK = 6
ON_OFFSET = 0


def _counts(gt_tr: np.ndarray, pr_tr: np.ndarray) -> tuple[float, float, float, int]:
    tp = int((gt_tr & pr_tr).sum())
    fp = int((~gt_tr & pr_tr).sum())
    fn = int((gt_tr & ~pr_tr).sum())
    pre = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * pre * rec / (pre + rec) if (tp + fp) and (tp + fn) and (pre + rec) > 0 else float("nan")
    return pre, rec, f1, tp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="openloop_mse.py --dump_hand_logits 의 출력")
    ap.add_argument("--split", default="val", help="임계값을 고를 스플릿 (기본 val)")
    ap.add_argument("--min_precision", type=float, default=0.60,
                    help="이 정밀도 이상 중 재현율이 가장 큰 임계값을 고른다 (기본 0.60)")
    ap.add_argument("--grid", type=float, nargs=3, default=[-2.0, 6.0, 0.25],
                    metavar=("START", "STOP", "STEP"), help="임계값 격자")
    ap.add_argument("--out", default=None, help="고른 임계값을 json 으로 저장")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    logits = d["logits"]                                   # (N, H, 6*손개수)
    hands = [str(h) for h in d["hands"]]
    split = np.array([str(s) for s in d["split"]])
    if logits.shape[-1] != HAND_BLOCK * len(hands):
        print(f"[오류] 로짓 폭 {logits.shape[-1]} 이 손 {len(hands)}개 x {HAND_BLOCK} 와 다르다")
        return 1

    splits = sorted(set(split.tolist()))
    if args.split not in splits:
        print(f"[오류] 스플릿 {args.split!r} 이 없다. 있는 것: {splits}")
        return 1

    ths = np.arange(args.grid[0], args.grid[1] + 1e-9, args.grid[2])
    chosen: dict[str, float] = {}
    report: dict[str, dict] = {}

    for hi, hand in enumerate(hands):
        on_logit = logits[..., hi * HAND_BLOCK + ON_OFFSET]      # (N, H)
        g_on = d[f"gt_on/{hand}"].astype(bool)
        c_on = d[f"cur_on/{hand}"].astype(bool)
        gt_tr = g_on != c_on

        print(f"\n=== {hand} ===  전이율 {gt_tr.mean():.4f}  (윈도 {gt_tr.shape[0]}, 지평 {gt_tr.shape[1]})")
        print(f"{'임계값':>8s}{'정밀도':>10s}{'재현율':>10s}{'F1':>10s}{'TP':>8s}"
              f"{'정밀도(그외)':>14s}{'재현율(그외)':>14s}")

        sel_mask = split == args.split
        oth_mask = ~sel_mask
        rows = []
        for th in ths:
            pr_on = on_logit > th
            pr_tr = pr_on != c_on
            pre, rec, f1, tp = _counts(gt_tr[sel_mask], pr_tr[sel_mask])
            o_pre, o_rec, _, _ = _counts(gt_tr[oth_mask], pr_tr[oth_mask]) if oth_mask.any() \
                else (float("nan"), float("nan"), float("nan"), 0)
            rows.append({"th": float(th), "precision": pre, "recall": rec, "f1": f1, "tp": tp,
                         "precision_other": o_pre, "recall_other": o_rec})
            print(f"{th:8.2f}{pre:10.3f}{rec:10.3f}{f1:10.3f}{tp:8d}{o_pre:14.3f}{o_rec:14.3f}")

        # 정밀도 하한을 넘는 것 중 재현율 최대. 없으면 정밀도가 가장 높은 쪽.
        ok = [r for r in rows if not np.isnan(r["precision"]) and r["precision"] >= args.min_precision
              and r["tp"] > 0]
        if ok:
            best = max(ok, key=lambda r: (r["recall"], r["precision"]))
            why = f"정밀도>={args.min_precision} 중 재현율 최대"
        else:
            cand = [r for r in rows if not np.isnan(r["precision"]) and r["tp"] > 0]
            if not cand:
                print(f"  [경고] {hand}: 어느 임계값에서도 전이를 하나도 못 맞혔다 — 임계값 선택 불가")
                continue
            best = max(cand, key=lambda r: r["precision"])
            why = f"정밀도 {args.min_precision} 를 넘는 지점이 없어 정밀도 최대 지점"
        chosen[hand] = best["th"]
        report[hand] = {"chosen": best, "why": why, "transition_rate": float(gt_tr.mean()),
                        "grid": rows}
        print(f"  -> 선택 {best['th']:.2f}  정밀도 {best['precision']:.3f} "
              f"재현율 {best['recall']:.3f}  ({why})")

    print("\n임계값 (클라이언트에서 hand_logits 에 적용):")
    print("  " + json.dumps(chosen))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"chosen": chosen, "split": args.split,
                       "min_precision": args.min_precision, "detail": report}, f, indent=2)
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
