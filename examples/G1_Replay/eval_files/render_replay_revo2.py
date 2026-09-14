"""MuJoCo kinematic replay 렌더 v2 — G1 + BrainCo Revo2 5지 핸드 (좌 GT / 우 예측).

v1(render_replay.py, 손 중립 고정) 대비: 공식 revo2_description URDF를 손목에
장착한 합성 씬(examples/G1_Washer/assets/g1_revo2_scene.xml)으로 손 개폐까지 재생.

손 채널 매핑 (2026-08-15 가설 — unitree-g1-brainco-hand SetMotorMulti positions[6] 기준):
  dim0 → 엄지 굽힘(proximal), dim1 → 모드(재생 안 함), dim2 → 엄지 회전(metacarpal),
  dim3~6 → 검지/중지/약지/소지 proximal. 왼손은 부호 미러라 절대값 사용.
  distal 관절은 언더액추에이션 근사로 proximal에 비례(1.15배).

실행 (lerobot env):
  MUJOCO_GL=egl /home/chyoo/anaconda3/envs/lerobot/bin/python \
      examples/G1_Washer/eval_files/render_replay_revo2.py \
      --npz playground/Checkpoints/g1washer_v0_2026-08-14/pred_traj_s6000.npz \
      --out_dir playground/Checkpoints/g1washer_v0_2026-08-14/replay_videos_revo2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "assets/g1_revo2_scene.xml"

ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]
LEG_SDK = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

# 45D 레이아웃의 손 블록: L_hand 14:21, R_hand 21:28
# (채널: 0 파지요약/엄지굽힘, 1 모드, 2 엄지회전, 3~6 손가락)
HAND_CHAN_MAX = {0: 1.0, 2: 1.75, 3: 1.57, 4: 1.75, 5: 1.57, 6: 1.75}
FINGERS = ["index", "middle", "ring", "pinky"]


def body_channel_names():
    names = [None, None]                             # head 0:2
    names += [f"left_{j}_joint" for j in ARM_JOINTS]  # L_arm 2:9
    names += [None] * 7                               # L_hand 9:16
    names += [f"left_{j}_joint" for j in LEG_SDK]     # L_leg 16:22
    names += [f"right_{j}_joint" for j in LEG_SDK]    # R_leg 22:28
    names += [f"right_{j}_joint" for j in ARM_JOINTS] # R_arm 28:35
    names += [None] * 7                               # R_hand 35:42
    names += WAIST                                    # waist 42:45
    return names


def make_video_writer(out_path: Path | str, width: int, height: int, fps: int = 25):
    import av
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
    ap.add_argument("--size", type=int, nargs=2, default=[560, 560])
    ap.add_argument("--base_z", type=float, default=0.793)
    args = ap.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    W, Hpx = args.size
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, Hpx)

    def qadr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"joint not found: {name}"
        return model.jnt_qposadr[jid], model.jnt_range[jid]

    ch_qpos = np.full(45, -1)
    for i, nm in enumerate(body_channel_names()):
        if nm is not None:
            ch_qpos[i], _ = qadr(nm)
    active = ch_qpos >= 0

    # 손 매핑 테이블: (채널 인덱스, qpos 주소, joint max, 채널 max)
    hand_map = []
    for side, base, pfx in [("left", 9, "rvL_left"), ("right", 35, "rvR_right")]:
        a, r = qadr(f"{pfx}_thumb_metacarpal_joint"); hand_map.append((base + 2, a, r[1], HAND_CHAN_MAX[2]))
        a, r = qadr(f"{pfx}_thumb_proximal_joint");   hand_map.append((base + 0, a, r[1], HAND_CHAN_MAX[0]))
        ad, rd = qadr(f"{pfx}_thumb_distal_joint")
        hand_map.append((base + 0, ad, rd[1], HAND_CHAN_MAX[0]))  # distal ~ proximal 비례
        for fi, f in enumerate(FINGERS):
            ch = base + 3 + fi
            a, r = qadr(f"{pfx}_{f}_proximal_joint"); hand_map.append((ch, a, r[1], HAND_CHAN_MAX[3 + fi]))
            ad, rd = qadr(f"{pfx}_{f}_distal_joint"); hand_map.append((ch, ad, rd[1], HAND_CHAN_MAX[3 + fi]))

    renderer = mujoco.Renderer(model, height=Hpx, width=W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 160, -18, 2.4
    cam.lookat[:] = [0.25, 0.0, 0.85]

    # ground-snap 기준: 직립 기본 자세에서 발목 롤 링크의 z 높이
    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
                   for s in ("left", "right")]
    _d0 = mujoco.MjData(model)
    _d0.qpos[:] = model.qpos0; _d0.qpos[0:3] = [0, 0, args.base_z]
    mujoco.mj_forward(model, _d0)
    FOOT_REF_Z = min(_d0.xpos[b][2] for b in foot_bodies)

    from PIL import Image, ImageDraw

    def set_pose(data, q45, quat, label):
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = [0, 0, args.base_z]
        data.qpos[3:7] = quat
        data.qpos[ch_qpos[active]] = q45[active]
        for ch, adr, jmax, cmax in hand_map:
            v = abs(float(q45[ch])) / cmax  # 왼손 음수 미러 → 절대값 정규화
            data.qpos[adr] = min(v, 1.0) * jmax
        mujoco.mj_forward(model, data)
        # ground-snap
        foot_z = min(data.xpos[b][2] for b in foot_bodies)
        data.qpos[2] += FOOT_REF_Z - foot_z
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([8, 8, 8 + 11 * len(label) + 10, 34], fill=(0, 0, 0, 180))
        dr.text((14, 12), label, fill=(255, 255, 255))
        return np.asarray(img)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.npz)
    keys = sorted({k.rsplit("/", 1)[0] for k in z.files})
    for key in keys:
        gt, pred, st = z[f"{key}/gt"], z[f"{key}/pred"], z[f"{key}/state"]
        quats = st[:, 3:7] if st.shape[1] == 84 else st[:, 61:65]
        d_gt, d_pr = mujoco.MjData(model), mujoco.MjData(model)
        step = max(1, 50 // args.fps)
        vw = make_video_writer(out_dir / f"{key}_replay_revo2.mp4", width=W * 2, height=Hpx, fps=args.fps)
        for t in range(0, len(gt), step):
            fr = np.hstack([set_pose(d_gt, gt[t], quats[t], "GT"),
                            set_pose(d_pr, pred[t], quats[t], "PRED")])
            vw.append_data(fr)
        vw.close()
        print(f"saved {key}_replay_revo2.mp4 (좌 GT / 우 예측, Revo2 손 포함)")


if __name__ == "__main__":
    main()
