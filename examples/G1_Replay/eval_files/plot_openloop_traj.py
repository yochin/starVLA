"""연속 청크-연결 open-loop 궤적 플롯 (Isaac-GR00T open_loop_eval.py 방식 이식).

에피소드를 처음부터 따라가며 horizon(8)마다 재추론(관측은 GT 주입),
예측 청크를 시간축으로 이어붙여 GT 명령과 차원별로 겹쳐 그린다.
빨간 점 = 재추론 시점. 청크 경계의 불연속(실기 jerk 리스크)도 여기서 보인다.

※ 용도 한정: 태스크별 대표 에피소드의 한 구간만 그리므로 **정성 진단 전용**이다.
정량 근거는 openloop_mse.py(전 val 균등 윈도, macro/micro 분해)만 쓴다.

실행 (starvla_qwen35 env, 학습 종료 후 GPU 4 또는 5):
  CUDA_VISIBLE_DEVICES=4 python examples/G1_Washer/eval_files/plot_openloop_traj.py \
      --ckpt playground/Checkpoints/g1washer_v0_2026-08-14/checkpoints/steps_6000_pytorch_model.pt \
      --out_dir playground/Checkpoints/g1washer_v0_2026-08-14/traj_plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# openloop_mse.py의 상수·헬퍼 재사용
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openloop_mse import (  # noqa: E402
    DATA_ROOT, _DEFAULT_DATASETS, PARTS, VIEWS, H,
    decode_frames, load_episode,
)

# 45D 레이아웃에서 대표 차원 라벨 (플롯 y라벨용)
DIM_LABELS = (
    [f"head{i}" for i in range(2)]
    + [f"L_arm{i}" for i in range(7)]
    + [f"L_hand{i}" for i in range(7)]
    + [f"legs{i}" for i in range(12)]
    + [f"R_arm{i}" for i in range(7)]
    + [f"R_hand{i}" for i in range(7)]
    + [f"waist{i}" for i in range(3)]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", choices=["val", "train", "both"], default="val")
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--num_chunks", type=int, default=60, help="이어붙일 청크 수 (60×16=960스텝=19.2초)")
    ap.add_argument("--episodes_per_task", type=int, default=1)
    ap.add_argument("--no_send_state", action="store_true",
                    help="Do not include proprioceptive state in QwenOFT requests")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import random
    import torch
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    def _seed_everything(seed: int):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    _seed_everything(7)
    wrapper = PolicyServerWrapper(ckpt_path=args.ckpt, device="cuda", use_bf16=True)
    _seed_everything(7)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, split_type, _ in _DEFAULT_DATASETS:
        if args.split != "both" and split_type != args.split:
            continue
        ds_dir = (DATA_ROOT / ds_name).resolve()
        if not ds_dir.exists():
            continue
        eps = [json.loads(l) for l in open(ds_dir / "meta" / "episodes.jsonl")]
        tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
                 for l in open(ds_dir / "meta" / "tasks.jsonl")}
        default_task_text = tasks.get(0, "")

        for ep in eps[: args.episodes_per_task]:
            ei, T = ep["episode_index"], ep["length"]
            task_text = ep["tasks"][0] if (isinstance(ep.get("tasks"), list) and ep["tasks"]) else tasks.get(ep.get("task_index", 0), default_task_text)
            act45, st_model, s_raw = load_episode(ds_dir, ei)
            n_chunks = min(args.num_chunks, (T - args.start - H) // H)
            starts = [args.start + k * H for k in range(n_chunks)]
            frames = decode_frames(ds_dir, ei, starts)

            preds = []
            for wi, t in enumerate(starts):
                ex = dict(image=[frames[v][wi] for v in VIEWS], lang=task_text)
                if not args.no_send_state:
                    ex["state"] = st_model[t][None, :]
                out = wrapper.predict_action(examples=[ex], state_is_normalized=False)
                preds.append(np.asarray(out["actions"], dtype=np.float32)[0])  # (H, 45)
            preds_arr = np.stack(preds, axis=0)                                # (n_chunks, H, 45)
            pred_traj = np.concatenate(preds, axis=0)                          # (n*H, 45)
            gt_traj = act45[args.start: args.start + n_chunks * H]             # (n*H, 45)

            # 1. 45개 관절별 궤적 플롯
            mse = float(((pred_traj - gt_traj) ** 2).mean())
            split_label = "UNSEEN_VAL" if split_type == "val" else "SEEN_TRAIN"
            fig, axes = plt.subplots(nrows=45, ncols=1, figsize=(12, 90))
            fig.suptitle(f"{ds_name} [{split_label}] ep{ei:03d} — 연속 Open-Loop 궤적 (H={H}, {n_chunks}청크, 전체 MSE {mse:.5f})", y=0.9995, fontsize=12)
            xs = np.arange(len(gt_traj))
            for d in range(45):
                ax = axes[d]
                ax.plot(xs, gt_traj[:, d], label="GT (Ground Truth)", color="#1f77b4", lw=1.2)
                ax.plot(xs, pred_traj[:, d], label="PRED (16-step Chunks)", color="#ff7f0e", lw=1.0)
                for k in range(n_chunks):
                    # 추론 시점 수직 점선 및 마커
                    ax.axvline(k * H, color="red", linestyle="--", alpha=0.35, lw=0.8)
                    ax.plot(k * H, gt_traj[k * H, d], "ro", ms=3.0, label="추론 시점 (Chunk Start)" if (k == 0 and d == 0) else "")
                    if k % 2 == 0:
                        ax.axvspan(k * H, min((k + 1) * H, len(gt_traj)), color="gray", alpha=0.06)
                ax.set_ylabel(DIM_LABELS[d], fontsize=7, rotation=0, labelpad=25)
                ax.grid(True, linestyle=":", alpha=0.3)
                if d == 0:
                    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            plt.tight_layout()
            prefix = f"{ds_name.replace('/', '_')}_ep{ei:03d}"
            fp = out_dir / f"{prefix}_traj.png"
            plt.savefig(fp, dpi=80)
            plt.close()
            print(f"saved {fp}  (MSE {mse:.5f})")

            # 2. Action Horizon Step별 오차 추이 (Horizon Error Drift: Step 0 -> 15)
            # 청크 내 시간 경과에 따른 오차 변화 시각화
            gt_chunks = gt_traj.reshape(n_chunks, H, 45)
            step_err2 = ((preds_arr - gt_chunks) ** 2).mean(axis=(0, 2))  # (H,)
            fig_h, ax_h = plt.subplots(figsize=(10, 4.5))
            steps = np.arange(H)
            ax_h.plot(steps, step_err2, marker="o", color="#d62728", lw=2, label="전체 45D 평균 MSE")
            # 주요 부위별 곡선
            for pname, (a, b) in PARTS.items():
                p_err2 = ((preds_arr[..., a:b] - gt_chunks[..., a:b]) ** 2).mean(axis=(0, 2))
                ax_h.plot(steps, p_err2, linestyle="--", alpha=0.7, label=f"{pname} ({a}:{b})")
            ax_h.set_title(f"{ds_name} [{split_label}] ep{ei:03d} — Action Horizon Step별 오차 변화 (추론 직후 h=0 ~ 청크 끝 h={H-1})", fontsize=11)
            ax_h.set_xlabel(f"Action Horizon Step (0 ~ {H-1}) [0스텝={0.0:.2f}s ~ {H-1}스텝={(H-1)*0.02:.2f}s]")
            ax_h.set_ylabel("Mean Squared Error (MSE)")
            ax_h.set_xticks(steps)
            ax_h.grid(True, linestyle="--", alpha=0.5)
            ax_h.legend(fontsize=8, loc="upper left")
            plt.tight_layout()
            fp_h = out_dir / f"{prefix}_horizon_drift.png"
            plt.savefig(fp_h, dpi=100)
            plt.close()
            print(f"saved {fp_h}  (Horizon Drift: Step 0={step_err2[0]:.5f} -> Step {H-1}={step_err2[-1]:.5f})")


if __name__ == "__main__":
    main()
