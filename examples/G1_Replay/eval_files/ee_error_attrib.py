"""손끝(end-effector) 위치 오차와 관절별 기여도 분해.

관절 각도 오차는 링크 체인을 타고 증폭된다 — 어깨 1도와 손목 1도는 손끝
위치에 전혀 다른 크기로 나타난다. 이 스크립트는 MuJoCo FK로 실제 손끝
위치 오차를 재고, "그 관절만 GT로 되돌리면 오차가 얼마나 줄어드는가"로
기여도를 귀속시킨다(leave-one-out).

전제: perjoint_mse.py가 저장한 .npz (pred/gt 관절각, 45D 레이아웃).
씬은 render_replay_revo2.py와 동일한 g1_revo2_scene.xml을 쓴다.

실행:
  python examples/G1_Washer/eval_files/ee_error_attrib.py \
      --npz playground/Checkpoints/g1washer_v0_2026-08-14/perjoint_s6000.npz \
      --out playground/Checkpoints/g1washer_v0_2026-08-14/ee_attrib.json
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

from render_replay_revo2 import body_channel_names  # noqa: E402

SCENE = Path(__file__).resolve().parents[1] / "assets/g1_revo2_scene.xml"

CHAIN_RIGHT = {
    "R_arm.shoulder_pitch": 28, "R_arm.shoulder_roll": 29, "R_arm.shoulder_yaw": 30,
    "R_arm.elbow": 31, "R_arm.wrist_roll": 32, "R_arm.wrist_pitch": 33, "R_arm.wrist_yaw": 34,
    "waist.yaw": 42, "waist.roll": 43, "waist.pitch": 44,
}
CHAIN_LEFT = {
    "L_arm.shoulder_pitch": 2, "L_arm.shoulder_roll": 3, "L_arm.shoulder_yaw": 4,
    "L_arm.elbow": 5, "L_arm.wrist_roll": 6, "L_arm.wrist_pitch": 7, "L_arm.wrist_yaw": 8,
    "waist.yaw": 42, "waist.roll": 43, "waist.pitch": 44,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--max_windows", type=int, default=300)
    args = ap.parse_args()

    import mujoco

    z = np.load(args.npz, allow_pickle=True)
    pred, gt = z["pred"], z["gt"]           # (N, H, 45)
    N = min(args.max_windows, len(pred))
    pred, gt = pred[:N], gt[:N]

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    names = body_channel_names()
    qadr = {}
    for i, nm in enumerate(names):
        if nm is None:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if jid >= 0:
            qadr[i] = model.jnt_qposadr[jid]
    print(f"매핑된 관절 채널 {len(qadr)}개")

    ee_body = f"{args.hand}_wrist_yaw_link"
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
    assert bid >= 0, f"body not found: {ee_body}"
    print(f"손끝 기준 body: {ee_body}\n")

    def fk(vec45):
        data.qpos[:] = model.qpos0
        for i, adr in qadr.items():
            data.qpos[adr] = vec45[i]
        mujoco.mj_forward(model, data)
        return data.xpos[bid].copy()

    # 전체 프레임 평면화 (N*H)
    P = pred.reshape(-1, 45)
    G = gt.reshape(-1, 45)
    M = len(P)

    ee_gt = np.stack([fk(G[k]) for k in range(M)])
    ee_pr = np.stack([fk(P[k]) for k in range(M)])
    err_full = np.linalg.norm(ee_pr - ee_gt, axis=1)

    print(f"=== 손끝 위치 오차 ({args.hand} hand, {M} 프레임) ===")
    print(f"  평균 {err_full.mean()*100:.2f} cm   중앙값 {np.median(err_full)*100:.2f} cm   "
          f"p90 {np.percentile(err_full,90)*100:.2f} cm   최대 {err_full.max()*100:.2f} cm")

    # leave-one-out: 해당 관절만 GT로 되돌렸을 때 남는 오차
    chain = CHAIN_RIGHT if args.hand == "right" else CHAIN_LEFT
    rows = []
    for jname, d in chain.items():
        Pf = P.copy()
        Pf[:, d] = G[:, d]
        ee_fix = np.stack([fk(Pf[k]) for k in range(M)])
        err_fix = np.linalg.norm(ee_fix - ee_gt, axis=1)
        drop = err_full.mean() - err_fix.mean()
        rows.append(dict(joint=jname, dim=d,
                         residual_cm=float(err_fix.mean() * 100),
                         drop_cm=float(drop * 100),
                         share=float(drop / err_full.mean())))
    rows.sort(key=lambda r: -r["drop_cm"])

    print(f"\n=== 관절별 기여도 (그 관절만 GT로 되돌렸을 때) ===")
    print(f"{'joint':<24}{'남는 오차':>10}{'감소':>9}{'기여율':>8}")
    for r in rows:
        print(f"{r['joint']:<24}{r['residual_cm']:>9.2f}cm{r['drop_cm']:>8.2f}cm{r['share']*100:>7.1f}%")

    # 팔 전체 / 허리 전체를 각각 GT로
    arm_label = "오른팔 7개 전부" if args.hand == "right" else "왼팔 7개 전부"
    arm_dims = list(range(28, 35)) if args.hand == "right" else list(range(2, 9))
    waist_dims = list(range(42, 45))

    for label, dims in [(arm_label, arm_dims), ("허리 3개 전부", waist_dims)]:
        Pf = P.copy(); Pf[:, dims] = G[:, dims]
        e = np.linalg.norm(np.stack([fk(Pf[k]) for k in range(M)]) - ee_gt, axis=1)
        print(f"  [{label}] 남는 오차 {e.mean()*100:.2f} cm  (감소 {(err_full.mean()-e.mean())*100:.2f} cm)")

    json.dump({"n_frames": int(M), "hand": args.hand,
               "ee_err_cm": {"mean": float(err_full.mean()*100),
                             "median": float(np.median(err_full)*100),
                             "p90": float(np.percentile(err_full,90)*100),
                             "max": float(err_full.max()*100)},
               "attribution": rows}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
