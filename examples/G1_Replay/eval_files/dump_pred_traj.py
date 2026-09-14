"""연속 open-loop 예측 궤적을 npz로 덤프 (렌더링용 — plot_openloop_traj와 동일 방식).

실행 (starvla_qwen35 env, GPU 1장):
  CUDA_VISIBLE_DEVICES=4 python examples/G1_Washer/eval_files/dump_pred_traj.py \
      --ckpt playground/Checkpoints/g1washer_v0_2026-08-14/checkpoints/steps_6000_pytorch_model.pt \
      --out playground/Checkpoints/g1washer_v0_2026-08-14/pred_traj_s6000.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openloop_mse import DATA_ROOT, _DEFAULT_DATASETS, VIEWS, H, decode_frames, load_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["val", "train", "both"], default="both",
                    help="덤프 대상: val (검증용 unseen), train (학습용 seen), both (둘 다)")
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--num_chunks", type=int, default=60)
    ap.add_argument("--episodes_per_task", type=int, default=1)
    ap.add_argument("--ensemble", action="store_true", default=False,
                    help="Temporal ensembling 적용 (재추론 주기마다 청크를 중첩 블렌딩)")
    ap.add_argument("--ensemble_stride", type=int, default=4,
                    help="Ensemble 재추론 주기 스텝 수 (기본: 4스텝 = 50Hz 기준 80ms 주기)")
    ap.add_argument("--ensemble_mode", choices=["exp", "cosine", "linear", "uniform"], default="exp",
                    help="Ensembling 가중치 모드 (기본: exp)")
    ap.add_argument("--weight_decay", type=float, default=0.05,
                    help="exp 모드 지수 감쇠 계수")
    ap.add_argument("--no_send_state", action="store_true",
                    help="Do not include proprioceptive state in QwenOFT requests")
    args = ap.parse_args()

    import random
    import torch
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.tools.ensembler import TemporalEnsembler

    def _seed(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    _seed(7)
    wrapper = PolicyServerWrapper(ckpt_path=args.ckpt, device="cuda", use_bf16=True)
    _seed(7)

    out = {}
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
            total_steps = n_chunks * H

            if args.ensemble:
                stride = max(1, args.ensemble_stride)
                starts = [args.start + k * stride for k in range((total_steps) // stride)]
                frames = decode_frames(ds_dir, ei, starts)
                ensembler = TemporalEnsembler(
                    action_horizon=H,
                    mode=args.ensemble_mode,
                    weight_decay=args.weight_decay,
                    discrete_dims=[10, 36],
                )
                chunks_dict = {}
                for wi, t in enumerate(starts):
                    ex = dict(image=[frames[v][wi] for v in VIEWS], lang=task_text)
                    if not args.no_send_state:
                        ex["state"] = st_model[t][None, :]
                    o = wrapper.predict_action(examples=[ex], state_is_normalized=False)
                    chunks_dict[t] = np.asarray(o["actions"], dtype=np.float32)[0]
                pred_traj = ensembler.ensemble_trajectory(chunks_dict, start_step=args.start, num_steps=total_steps)
            else:
                starts = [args.start + k * H for k in range(n_chunks)]
                frames = decode_frames(ds_dir, ei, starts)
                preds = []
                for wi, t in enumerate(starts):
                    ex = dict(image=[frames[v][wi] for v in VIEWS], lang=task_text)
                    if not args.no_send_state:
                        ex["state"] = st_model[t][None, :]
                    o = wrapper.predict_action(examples=[ex], state_is_normalized=False)
                    preds.append(np.asarray(o["actions"], dtype=np.float32)[0])
                pred_traj = np.concatenate(preds, axis=0)

            key = f"{ds_name.replace('/', '_')}_ep{ei:03d}"
            sl = slice(args.start, args.start + total_steps)
            out[f"{key}/pred"] = pred_traj                       # (n*H, 45)
            out[f"{key}/gt"] = act45[sl]                         # (n*H, 45)
            out[f"{key}/state"] = s_raw[sl]                      # (n*H, raw_dim) — base quat 포함
            out[f"{key}/start"] = np.array(args.start)
            out[f"{key}/split"] = np.array(split_type)
            ens_str = f"ensemble=True (stride={args.ensemble_stride})" if args.ensemble else "ensemble=False"
            print(f"{key} [{split_type.upper()}]: {n_chunks} chunks ({ens_str})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
