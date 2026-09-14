"""관절 단위 open-loop 오차 분해 — openloop_mse.py의 부위 단위를 관절까지 쪼갠다.

openloop_mse.py와 동일한 윈도 추출·추론 경로를 재사용하고, 요약만 관절별로 낸다.
예측 배열을 .npz로 저장해 이후 분석(궤적 플롯 등)에 재추론 없이 쓸 수 있게 한다.

관절 이름은 SONIC의 g1_29dof_sonic_model12.yaml 순서를 따른다
(shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw).

실행:
  python examples/G1_Washer/eval_files/perjoint_mse.py \
      --ckpt playground/Checkpoints/g1washer_v0_2026-08-14/checkpoints/steps_6000_pytorch_model.pt \
      --stride 100 --max_windows_per_ep 12 --out .../perjoint.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openloop_mse import (  # noqa: E402
    H, MODE_DIMS, collect_windows, eval_model,
)

# 45D 레이아웃의 관절 이름 (= G1WasherDataConfig.action_keys concat 순서)
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
       "wrist_roll", "wrist_pitch", "wrist_yaw"]
HAND = ["h0", "MODE", "thumb", "f3", "f4", "f5(=f3)", "f6(=f4)"]
# 다리는 Unitree SDK 순서(hip_pitch, hip_roll, hip_yaw)다. SONIC의
# g1_29dof_sonic_model12.yaml은 관절 인덱스 맵을 yaw/roll/pitch로 적어두었지만
# 같은 파일의 DEFAULT_DOF_ANGLES 주석은 pitch/roll/yaw이라 서로 충돌한다.
# 데이터로 판정(2026-08-18): dim0이 knee와 상관 -0.800, 좌우 동상 +0.839 →
# 웅크릴 때 무릎과 연동되는 hip_pitch가 맞다(yaw는 좌우 역상·소진폭).
LEG = [f"{s}_{j}" for s in ("L", "R")
       for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
JOINTS = (["head.0", "head.1"]
          + [f"L_arm.{j}" for j in ARM]
          + [f"L_hand.{j}" for j in HAND]
          + [f"legs.{j}" for j in LEG]
          + [f"R_arm.{j}" for j in ARM]
          + [f"R_hand.{j}" for j in HAND]
          + ["waist.yaw", "waist.roll", "waist.pitch"])
assert len(JOINTS) == 45, len(JOINTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "train", "both"], default="both",
                    help="평가 대상 분할: val (미학습 검증용), train (학습 데이터셋), both (둘 다 비교)")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--max_windows_per_ep", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_send_state", action="store_true",
                    help="Do not include proprioceptive state in QwenOFT requests")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    windows = collect_windows(args.stride, args.max_windows_per_ep, target_split=args.split)
    print(f"\n총 {len(windows)} windows (split={args.split})\n")

    gt = np.stack([w["gt"] for w in windows])            # (N, H, 45)
    persist = np.stack([w["persist"] for w in windows])  # (N, H, 45)
    pred = eval_model(
        args.ckpt,
        windows,
        args.batch_size,
        "trained",
        send_state=not args.no_send_state,
    )

    split_types = np.array([w["split_type"] for w in windows])
    ds_names = np.array([w["ds_name"] for w in windows])

    out_npz = Path(args.out).with_suffix(".npz")
    np.savez_compressed(out_npz, pred=pred, gt=gt, persist=persist,
                        ds_name=ds_names, split_type=split_types,
                        ep=np.array([w["ep"] for w in windows]),
                        t=np.array([w["t"] for w in windows]))
    print(f"arrays -> {out_npz}")

    def compute_joint_rows(mask):
        rows = []
        p_sub, g_sub, per_sub = pred[mask], gt[mask], persist[mask]
        for d in range(45):
            e_pred = float(((p_sub[..., d] - g_sub[..., d]) ** 2).mean())
            e_per = float(((per_sub[..., d] - g_sub[..., d]) ** 2).mean())
            rows.append(dict(
                dim=d, joint=JOINTS[d],
                mse_pred=e_pred, mse_persist=e_per,
                ratio=(e_pred / e_per if e_per > 1e-12 else float("inf")),
                mae_pred_deg=float(np.abs(p_sub[..., d] - g_sub[..., d]).mean() * 57.2958),
                gt_std=float(g_sub[..., d].std()),
                gt_range=float(g_sub[..., d].max() - g_sub[..., d].min()),
                is_mode=d in MODE_DIMS,
            ))
        return rows

    report = {"n_windows": len(windows), "ckpt": args.ckpt, "split": args.split}
    all_mask = np.ones(len(windows), dtype=bool)
    report["joints_overall"] = compute_joint_rows(all_mask)

    val_mask = split_types == "val"
    train_mask = split_types == "train"

    if val_mask.any():
        report["joints_val_unseen"] = compute_joint_rows(val_mask)
    if train_mask.any():
        report["joints_train_seen"] = compute_joint_rows(train_mask)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}\n")

    def print_table(title, rows):
        hdr = f"{'dim':>3} {'joint':<22} {'MSE':>10} {'기준선':>10} {'배율':>7} {'MAE(deg)':>9} {'GT std':>8} {'GT range':>9}"
        print(f"\n=== {title} ===")
        print(hdr); print("-" * len(hdr))
        for r in rows:
            mark = "  <MODE>" if r["is_mode"] else ""
            print(f"{r['dim']:>3} {r['joint']:<22} {r['mse_pred']:>10.5f} {r['mse_persist']:>10.5f} "
                  f"{r['ratio']:>7.2f} {r['mae_pred_deg']:>9.2f} {r['gt_std']:>8.3f} {r['gt_range']:>9.3f}{mark}")

    if val_mask.any():
        print_table("VAL (UNSEEN) 관절별 오차", report["joints_val_unseen"])
    if train_mask.any():
        print_table("TRAIN (SEEN) 관절별 오차", report["joints_train_seen"])


if __name__ == "__main__":
    main()
