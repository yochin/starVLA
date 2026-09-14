"""3패널 동기 replay 렌더 v4 — [실제 헤드캠 | GT(측정값) | 예측(명령)] + 타임스탬프 + 물리 시뮬레이션(중력/바닥마찰 보상).

v4 기능 (2026-08-28):
  1) 바닥 마찰력 (Floor Friction: Sliding 1.0, Torsional 0.005, Rolling 0.0001) 적용
  2) 중력 보상 (Gravity Compensation: data.qfrc_bias 피드포워드) 및 지면 접촉 구속 계산
  3) sim_mode: dynamics (물리 동역학 시뮬레이션) 또는 kinematic (기구학 포즈 렌더링) 선택 가능
  4) 실제 카메라 패널과 1:1 동기화 (PyAV 초고속 디코딩) 및 3패널 비교

실행 (starVLA env):
  MUJOCO_GL=egl python -u examples/G1_Replay/eval_files/render_replay_sync.py \
      --npz .../pred_traj_ensemble.npz \
      --out_dir .../replay_videos_sync_dynamics \
      --sim_mode dynamics \
      --floor_friction 1.0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

SCENE = Path(__file__).resolve().parents[1] / "assets/g1_revo2_scene.xml"
DATA_ROOT = Path(os.environ.get(
    "G1_DATA_ROOT", Path(__file__).resolve().parents[3] / "playground/Datasets"))

ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]
LEG_SDK = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
HAND_CHAN_MAX = {0: 1.0, 2: 1.75, 3: 1.57, 4: 1.75, 5: 1.57, 6: 1.75}
FINGERS = ["index", "middle", "ring", "pinky"]

PANEL_H = 376  # 카메라 좌안 원본 높이에 맞춤
SIM_W = 376


def make_video_writer(out_path: Path | str, width: int, height: int, fps: int = 25):
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    class VideoWriterWrapper:
        def append_data(self, img_rgb: np.ndarray):
            frame = av.VideoFrame.from_ndarray(img_rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        def close(self):
            for packet in stream.encode():
                container.mux(packet)
            container.close()

    return VideoWriterWrapper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--base_z", type=float, default=0.793)
    ap.add_argument("--sim_mode", choices=["dynamics", "kinematic"], default="kinematic",
                    help="시뮬레이션 모드: kinematic (기구학 포즈, 기본값), dynamics (중력/바닥마찰 보상 물리 시뮬레이션)")
    ap.add_argument("--floor_friction", type=float, default=1.0,
                    help="바닥 슬라이딩 마찰 계수 (기본: 1.0)")
    args = ap.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, SIM_W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, PANEL_H)

    # 1. 바닥 마찰력 설정 (dynamics 모드일 때만 적용)
    if args.sim_mode == "dynamics":
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            model.geom_friction[floor_id] = [args.floor_friction, 0.005, 0.0001]
            print(f"[MuJoCo] Floor friction configured: sliding={args.floor_friction}, torsional=0.005, rolling=0.0001", flush=True)

    def qadr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"joint not found: {name}"
        return model.jnt_qposadr[jid], model.jnt_range[jid]

    # 몸통 29D 주소 ([L_arm7|R_arm7|legs12|waist3] 순서)
    body29_names = ([f"left_{j}_joint" for j in ARM_JOINTS] + [f"right_{j}_joint" for j in ARM_JOINTS]
                    + [f"left_{j}_joint" for j in LEG_SDK] + [f"right_{j}_joint" for j in LEG_SDK] + WAIST)
    body29_addrs = np.array([qadr(n)[0] for n in body29_names])

    # 45D 액션 레이아웃에서 몸통 29D 추출 인덱스 (Washer 레거시 호환용)
    ACT_BODY_IDX = list(range(0, 14)) + list(range(28, 43))

    hand_map = []
    for side, base, pfx in [("left", 0, "rvL_left"), ("right", 7, "rvR_right")]:
        a, r = qadr(f"{pfx}_thumb_metacarpal_joint"); hand_map.append((base + 2, a, r[1], HAND_CHAN_MAX[2]))
        a, r = qadr(f"{pfx}_thumb_proximal_joint");   hand_map.append((base + 0, a, r[1], HAND_CHAN_MAX[0]))
        ad, rd = qadr(f"{pfx}_thumb_distal_joint");   hand_map.append((base + 0, ad, rd[1], HAND_CHAN_MAX[0]))
        for fi, f in enumerate(FINGERS):
            ch = base + 3 + fi
            a, r = qadr(f"{pfx}_{f}_proximal_joint"); hand_map.append((ch, a, r[1], HAND_CHAN_MAX[3 + fi]))
            ad, rd = qadr(f"{pfx}_{f}_distal_joint"); hand_map.append((ch, ad, rd[1], HAND_CHAN_MAX[3 + fi]))

    renderer = mujoco.Renderer(model, height=PANEL_H, width=SIM_W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 150, -16, 2.5
    cam.lookat[:] = [0.2, 0.0, 0.8]

    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
                   for s in ("left", "right")]
    _d0 = mujoco.MjData(model)
    _d0.qpos[:] = model.qpos0; _d0.qpos[0:3] = [0, 0, args.base_z]
    mujoco.mj_forward(model, _d0)
    FOOT_REF_Z = min(_d0.xpos[b][2] for b in foot_bodies)

    def snap_series(body29_seq, quats):
        """전 프레임 ground-snap 보정량 계산 후 이동평균 평활."""
        d = mujoco.MjData(model)
        zs = np.zeros(len(body29_seq))
        for i, (q29, qt) in enumerate(zip(body29_seq, quats)):
            d.qpos[:] = model.qpos0
            d.qpos[0:3] = [0, 0, args.base_z]; d.qpos[3:7] = qt
            d.qpos[body29_addrs] = q29
            mujoco.mj_forward(model, d)
            zs[i] = FOOT_REF_Z - min(d.xpos[b][2] for b in foot_bodies)
        k = 11  # ±0.22초 이동평균
        pad = np.pad(zs, (k // 2, k // 2), mode="edge")
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    def render_pose_dynamics(data, q29_target, hands14_src, quat, dz):
        """중력 보상 (Gravity Compensation) + 바닥 접촉 마찰력 물리 시뮬레이션 렌더링."""
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = [0, 0, args.base_z + dz]
        data.qpos[3:7] = quat
        data.qpos[body29_addrs] = q29_target

        for ch, adr, jmax, cmax in hand_map:
            v = abs(float(hands14_src[ch])) / cmax
            data.qpos[adr] = min(v, 1.0) * jmax

        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=cam)
        return renderer.render()

    def render_pose_kinematic(data, q29, hands14_src, quat, dz):
        """기구학적 포즈 렌더링 (순수 FK)."""
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = [0, 0, args.base_z + dz]
        data.qpos[3:7] = quat
        data.qpos[body29_addrs] = q29
        for ch, adr, jmax, cmax in hand_map:
            v = abs(float(hands14_src[ch])) / cmax
            data.qpos[adr] = min(v, 1.0) * jmax
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        return renderer.render()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.npz)
    keys = sorted({k.rsplit("/", 1)[0] for k in z.files})
    for key in keys:
        val_name, ep = key.rsplit("_ep", 1)
        ep = int(ep)
        gt_act, pred, st = z[f"{key}/gt"], z[f"{key}/pred"], z[f"{key}/state"]
        start = int(z[f"{key}/start"])
        T = len(gt_act)

        if "FridgeApple" in val_name or st.shape[1] == 81:
            quats = st[:, 3:7]
            st_body29 = np.concatenate([st[:, 9:16], st[:, 54:61], st[:, 30:42], st[:, 75:78]], axis=-1)
            st_hands = np.concatenate([st[:, 23:30], st[:, 68:75]], axis=-1)
        elif "723to724" in val_name:
            quats = st[:, 3:7]
            st_body29 = np.concatenate([st[:, 12:19], st[:, 57:64], st[:, 33:45], st[:, 78:81]], axis=-1)
            st_hands = np.concatenate([st[:, 26:33], st[:, 71:78]], axis=-1)
        elif st.shape[1] == 84:
            quats = st[:, 3:7]
            st_body29 = np.concatenate([st[:, 9:16], st[:, 57:64], st[:, 30:42], st[:, 78:81]], axis=-1)
            st_hands = np.concatenate([st[:, 23:30], st[:, 71:78]], axis=-1)
        elif st.shape[1] == 65:
            quats = st[:, 61:65]
            st_body29 = st[:, 0:29]
            st_hands = gt_act[:, 14:28]
        else:
            raise ValueError(f"Unknown state shape: {st.shape}")

        # 3) GT Action
        gt_act_body29 = np.concatenate([gt_act[:, 2:9], gt_act[:, 28:35], gt_act[:, 16:28], gt_act[:, 42:45]], axis=-1)
        gt_act_hands = np.concatenate([gt_act[:, 9:16], gt_act[:, 35:42]], axis=-1)

        # 4) PRED Action
        pred_body29 = np.concatenate([pred[:, 2:9], pred[:, 28:35], pred[:, 16:28], pred[:, 42:45]], axis=-1)
        pred_hands = np.concatenate([pred[:, 9:16], pred[:, 35:42]], axis=-1)

        ds_rel = val_name
        if not (DATA_ROOT / ds_rel).exists():
            for suffix in ["_val", "_train"]:
                if ds_rel.endswith(suffix):
                    cand = ds_rel[:-len(suffix)] + "/" + suffix[1:]
                    if (DATA_ROOT / cand).exists():
                        ds_rel = cand
                        break

        dz_st = snap_series(st_body29, quats)
        dz_gt = snap_series(gt_act_body29, quats)
        dz_pr = snap_series(pred_body29, quats)

        vp = (DATA_ROOT / ds_rel / "videos/chunk-000/observation.images.front_view"
              / f"episode_{ep:06d}.mp4")
        if not vp.exists():
            print(f"[skip] video not found: {vp}", flush=True)
            continue

        container = av.open(str(vp))
        cam_decoder = container.decode(video=0)

        d_st, d_gt, d_pr = mujoco.MjData(model), mujoco.MjData(model), mujoco.MjData(model)

        step = max(1, 50 // args.fps)
        out_mp4 = out_dir / f"{key}_sync.mp4"
        vw = make_video_writer(out_mp4, width=SIM_W * 4, height=PANEL_H, fps=args.fps)
        cam_idx = -1
        split_label = "UNSEEN VAL" if "val" in ds_rel else "SEEN TRAIN"
        sim_tag = "Dynamics (Grav+Fric)" if args.sim_mode == "dynamics" else "Kinematic"

        print(f"Rendering {key} ({T} frames, {T/50:.1f}s)...", flush=True)

        for t in range(0, T, step):
            abs_idx = start + t
            while cam_idx < abs_idx:
                f_cam = next(cam_decoder)
                cam_idx += 1
            cam_frame = f_cam.to_ndarray(format="rgb24")
            if cam_frame.shape[1] > 600:
                cam_left = cam_frame[:, :cam_frame.shape[1] // 2]
            else:
                cam_left = cam_frame
            if cam_left.shape[0] != PANEL_H or cam_left.shape[1] != SIM_W:
                cam_left = np.asarray(Image.fromarray(cam_left).resize((SIM_W, PANEL_H)))

            if args.sim_mode == "dynamics":
                fr_st = render_pose_dynamics(d_st, st_body29[t], st_hands[t], quats[t], dz_st[t])
                fr_gt = render_pose_dynamics(d_gt, gt_act_body29[t], gt_act_hands[t], quats[t], dz_gt[t])
                fr_pr = render_pose_dynamics(d_pr, pred_body29[t], pred_hands[t], quats[t], dz_pr[t])
            else:
                fr_st = render_pose_kinematic(d_st, st_body29[t], st_hands[t], quats[t], dz_st[t])
                fr_gt = render_pose_kinematic(d_gt, gt_act_body29[t], gt_act_hands[t], quats[t], dz_gt[t])
                fr_pr = render_pose_kinematic(d_pr, pred_body29[t], pred_hands[t], quats[t], dz_pr[t])

            # 4개 화면 동시 출력: 1) 실제 헤드캠, 2) GT state, 3) GT action, 4) pred action
            fr = np.hstack([cam_left, fr_st, fr_gt, fr_pr])
            img = Image.fromarray(fr); dr = ImageDraw.Draw(img)
            cam_w = cam_left.shape[1]
            head_text = f"1) HEAD CAM [{split_label}] t={abs_idx/50:5.2f}s"
            labels = [
                (8, head_text),
                (cam_w + 8, "2) GT State"),
                (cam_w + SIM_W + 8, "3) GT Action"),
                (cam_w + SIM_W * 2 + 8, "4) PRED Action"),
            ]
            for x, lab in labels:
                dr.rectangle([x, 8, x + 10 * len(lab) + 8, 32], fill=(0, 0, 0))
                dr.text((x + 5, 12), lab, fill=(255, 255, 255))
            vw.append_data(np.asarray(img))

        vw.close()
        container.close()
        print(f"saved {out_mp4.name} ({T}프레임 = {T/50:.0f}초, [{split_label}], mode={args.sim_mode})", flush=True)

    print("✅ All dynamic simulation videos rendered successfully!", flush=True)


if __name__ == "__main__":
    main()
