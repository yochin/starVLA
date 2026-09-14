"""Unitree G1 DexHands Open-Loop MSE 평가 — Train(Seen) vs Val(Unseen) 비교.

조건:
  1) trained    : 학습 체크포인트 (PolicyServerWrapper 경로)
  2) untrained  : step-0 체크포인트 (선택)
  3) persistence: zero-motion baseline — 직전 명령 action[t-1]을 16스텝 반복
                  (정지·저속 구간 기준선)

45D 레이아웃 (UnitreeG1DexHandsDirectGR00TDataConfig):
  [head 0:2 | L_arm 2:9 | L_hand 9:16 | legs 16:28 | R_arm 28:35 | R_hand 35:42 | waist 42:45]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path(os.environ.get(
    "G1_DATA_ROOT", REPO_ROOT / "playground/Datasets"))

# (dataset_name, split_type: 'val' [UNSEEN] or 'train' [SEEN], unnorm_key)
_DEFAULT_DATASETS = [
    ("FridgePickGrapes721SepStateObs/val", "val", "new_embodiment"),
    ("FridgePickGrapes723to724StateObs/val", "val", "new_embodiment"),
    ("FridgeApple/val", "val", "new_embodiment"),
    ("FridgePickGrapes721SepStateObs/train", "train", "new_embodiment"),
    ("FridgePickGrapes723to724StateObs/train", "train", "new_embodiment"),
    ("FridgeApple/train", "train", "new_embodiment"),
]

VIEWS = ["front_view", "left_wrist_view", "right_wrist_view"]

# 45D 레이아웃 파트 정의
PARTS = {
    "head": (0, 2),
    "L_arm": (2, 9),
    "L_hand": (9, 16),
    "legs": (16, 28),
    "R_arm": (28, 35),
    "R_hand": (35, 42),
    "waist": (42, 45),
}

# 손 dim1은 관절각이 아니라 {0,1,2,3} 이산 모드 셀렉터 (L_hand dim1=10, R_hand dim1=36)
MODE_DIMS = [10, 36]
CONT_DIMS = [d for d in range(45) if d not in MODE_DIMS]

H = int(os.environ.get("G1_ACTION_HORIZON", "16"))  # action horizon
EVAL_SEED = 7


def load_episode(ds_dir: Path, ep_idx: int):
    """parquet에서 action(45D)과 observation.state(84D 또는 81D)를 로드하고 모델용 state(81D)를 분리."""
    import pyarrow.parquet as pq
    pf = ds_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    tbl = pq.read_table(pf, columns=["observation.state", "action"]).to_pandas()
    s_raw = np.stack(tbl["observation.state"].values).astype(np.float32)
    act = np.stack(tbl["action"].values).astype(np.float32)               # (T, 45)
    # waist cmd echo [42:45] 제외 -> 81D
    if s_raw.shape[-1] == 84:
        st_model = np.concatenate([s_raw[:, 0:42], s_raw[:, 45:84]], axis=-1)
    else:
        st_model = s_raw
    return act, st_model, s_raw


def decode_frames(ds_dir: Path, ep_idx: int, frame_idxs: list[int]) -> dict[str, list[np.ndarray]]:
    """각 뷰 mp4를 순차 디코드하며 필요한 프레임만 수집."""
    import av
    want = set(frame_idxs)
    out = {}
    for view in VIEWS:
        vp = ds_dir / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{ep_idx:06d}.mp4"
        got = {}
        with av.open(str(vp)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i in want:
                    got[i] = frame.to_ndarray(format="rgb24")
                if len(got) == len(want):
                    break
        out[view] = [got[i] for i in frame_idxs]
    return out


def collect_windows(stride: int, max_windows_per_ep: int, target_split: str = "both"):
    """모든 에피소드에서 (윈도 메타, GT, persistence 예측, 원시 state) 수집."""
    windows = []
    for ds_name, split_type, unnorm_key in _DEFAULT_DATASETS:
        if target_split != "both" and split_type != target_split:
            continue
        ds_dir = (DATA_ROOT / ds_name).resolve()
        if not ds_dir.exists():
            print(f"[skip] {ds_dir} not found")
            continue
        eps = [json.loads(l) for l in open(ds_dir / "meta" / "episodes.jsonl")]
        tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
                 for l in open(ds_dir / "meta" / "tasks.jsonl")}
        default_task_text = tasks.get(0, "")
        for ep in eps:
            ei, T = ep["episode_index"], ep["length"]
            task_text = ep["tasks"][0] if (isinstance(ep.get("tasks"), list) and ep["tasks"]) else tasks.get(ep.get("task_index", 0), default_task_text)
            act45, st_model, s_raw = load_episode(ds_dir, ei)
            n_max = max(1, (T - 50 - H) // stride + 1)
            n = min(max_windows_per_ep, n_max)
            starts = np.unique(np.linspace(50, T - H - 1, num=n).astype(int)).tolist()
            if not starts:
                continue
            frames = decode_frames(ds_dir, ei, starts)
            tag = f"{ds_name} [{'UNSEEN_VAL' if split_type == 'val' else 'SEEN_TRAIN'}]"
            for wi, t in enumerate(starts):
                windows.append(dict(
                    ds_name=ds_name, split_type=split_type, unnorm_key=unnorm_key,
                    display_tag=tag, ep=ei, t=t,
                    lang=task_text,
                    gt=act45[t:t + H],                       # (H,45)
                    persist=np.tile(act45[t - 1], (H, 1)),   # (H,45)
                    state=st_model[t],                       # (81,)
                    images=[frames[v][wi] for v in VIEWS],   # 3 views (front_view will be stereo-split)
                ))
        print(f"[collect] {ds_name} ({split_type.upper()}): {len(eps)} eps -> 누적 {len(windows)} windows")
    return windows


def eval_model(ckpt_path: str, windows, batch_size: int, tag: str, send_state: bool = True):
    """PolicyServerWrapper로 예측 → 비정규화 액션 (N,H,45)."""
    import random
    import torch
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    def _seed_everything(seed: int):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    _seed_everything(EVAL_SEED)
    wrapper = PolicyServerWrapper(ckpt_path=ckpt_path, device="cuda", use_bf16=True)
    _seed_everything(EVAL_SEED)
    preds = np.zeros((len(windows), H, 45), dtype=np.float32)
    for s in range(0, len(windows), batch_size):
        chunk = list(range(s, min(s + batch_size, len(windows))))
        examples = [dict(image=windows[i]["images"], lang=windows[i]["lang"]) for i in chunk]
        if send_state:
            for example, i in zip(examples, chunk):
                example["state"] = windows[i]["state"][None, :]
        out = wrapper.predict_action(examples=examples, state_is_normalized=False)
        acts = np.asarray(out["actions"], dtype=np.float32)  # (B, H, 45)
        for j, i in enumerate(chunk):
            preds[i] = acts[j]
        if s % (batch_size * 5) == 0:
            print(f"[{tag}] {s + len(chunk)}/{len(windows)}")
    del wrapper
    torch.cuda.empty_cache()
    return preds


def summarize(pred, windows):
    gt = np.stack([w["gt"] for w in windows])          # (N,H,45)
    err2 = (pred - gt) ** 2
    err1 = np.abs(pred - gt)
    res = {
        "overall_continuous": float(err2[..., CONT_DIMS].mean()),
        "overall_mae_continuous": float(err1[..., CONT_DIMS].mean()),
        "hand_mode_acc": float(
            (np.clip(np.round(pred[..., MODE_DIMS]), 0, 3) == gt[..., MODE_DIMS]).mean()),
    }

    # Val vs Train 분리 집계
    splits = sorted({w["split_type"] for w in windows})
    for sp in splits:
        sp_mask = np.array([w["split_type"] == sp for w in windows])
        res[f"overall_continuous_{sp}"] = float(err2[sp_mask][..., CONT_DIMS].mean())
        res[f"overall_mae_continuous_{sp}"] = float(err1[sp_mask][..., CONT_DIMS].mean())

    # Task별 집계
    tags = sorted({w["display_tag"] for w in windows})
    task_mses_val, task_mses_train = [], []
    for tag in tags:
        m = np.array([w["display_tag"] == tag for w in windows])
        mse_val = float(err2[m][..., CONT_DIMS].mean())
        res[f"task/{tag}"] = mse_val
        if "[UNSEEN_VAL]" in tag:
            task_mses_val.append(mse_val)
        else:
            task_mses_train.append(mse_val)

    if task_mses_val:
        res["macro_val_unseen_mse"] = float(np.mean(task_mses_val))
    if task_mses_train:
        res["macro_train_seen_mse"] = float(np.mean(task_mses_train))

    for p, (a, b) in PARTS.items():
        dims = [d for d in range(a, b) if d not in MODE_DIMS]
        res[f"part/{p}"] = float(err2[..., dims].mean())
        res[f"mae/part/{p}"] = float(err1[..., dims].mean())

    for h in range(H):
        res[f"step/{h}"] = float(err2[:, h, CONT_DIMS].mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained_ckpt", required=True, nargs="+")
    ap.add_argument("--untrained_ckpt", default=None)
    ap.add_argument("--split", choices=["val", "train", "both"], default="both",
                    help="평가 대상 분할: val (미학습 검증용), train (학습 데이터셋), both (둘 다 비교)")
    ap.add_argument("--stride", type=int, default=100, help="윈도 간격 (프레임, 100=2초)")
    ap.add_argument("--max_windows_per_ep", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_send_state", action="store_true",
                    help="Do not include proprioceptive state in QwenOFT requests")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    windows = collect_windows(args.stride, args.max_windows_per_ep, target_split=args.split)
    print(f"\n총 {len(windows)} windows (split={args.split})\n")

    results = {"n_windows": len(windows), "stride": args.stride, "split": args.split,
               "layout": {k: list(v) for k, v in PARTS.items()}}

    persist = np.stack([w["persist"] for w in windows])
    results["persistence"] = summarize(persist, windows)
    print("persistence:", json.dumps(results["persistence"], indent=1)[:400])

    import re as _re
    trained_labels = []
    for ck in args.trained_ckpt:
        m = _re.search(r"steps_(\d+)", Path(ck).name)
        label = f"trained_{m.group(1)}" if m else f"trained_{Path(ck).stem}"
        trained_labels.append(label)
        results[label] = summarize(
            eval_model(ck, windows, args.batch_size, label, send_state=not args.no_send_state), windows)
        print(f"{label}:", json.dumps(results[label], indent=1)[:400])
    results["trained"] = results[trained_labels[-1]]

    if args.untrained_ckpt:
        results["untrained"] = summarize(
            eval_model(args.untrained_ckpt, windows, args.batch_size, "untrained",
                       send_state=not args.no_send_state), windows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")

    # 요약 표 출력
    conds = trained_labels + [c for c in ["untrained", "persistence"] if c in results]
    rows = []
    if "macro_val_unseen_mse" in results["trained"]:
        rows.append("macro_val_unseen_mse")
    if "macro_train_seen_mse" in results["trained"]:
        rows.append("macro_train_seen_mse")
    rows += ["overall_continuous", "overall_mae_continuous", "hand_mode_acc"]
    rows += [k for k in results["trained"].keys() if k.startswith("task/")]
    rows += [f"part/{p}" for p in PARTS]

    print(f"\n{'':38s}" + "".join(f"{c:>16s}" for c in conds))
    for row in rows:
        print(f"{row:38s}" + "".join(f"{results[c].get(row, 0.0):16.6f}" for c in conds))

    print("\n" + "=" * 65)
    print("=== Action Horizon Step별 오차 추이 (h=0: 추론 직후 ~ h=15: 청크 끝) ===")
    step_hdr = f"{'Horizon Step':>18} {'시간 (s)':>10}" + "".join(f"{c:>16s}" for c in conds)
    print(step_hdr)
    print("-" * len(step_hdr))
    for h in range(H):
        t_sec = h * 0.02
        row_str = f"Step {h:>2d} ({h+1:>2d}/{H}) {t_sec:>9.2f}s"
        row_str += "".join(f"{results[c].get(f'step/{h}', 0.0):16.6f}" for c in conds)
        print(row_str)


if __name__ == "__main__":
    main()
