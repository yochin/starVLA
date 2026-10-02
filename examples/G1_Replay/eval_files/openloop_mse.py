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

# 평가 대상 데이터셋 이름. 여기 없는 데이터셋은 DATA_ROOT 에 있어도 평가하지 않는다.
_DEFAULT_DATASET_NAMES = [
    "FridgePickGrapes721SepStateObs",
    "FridgePickGrapes723to724StateObs",
    "FridgeApple",
]


def _build_datasets(names: list[str]) -> list[tuple[str, str, str]]:
    """(dataset_name, split_type: 'val' [UNSEEN] or 'train' [SEEN], unnorm_key) 목록 생성."""
    return [
        (f"{name}/{split}", split, "new_embodiment")
        for split in ("val", "train")
        for name in names
    ]


# G1_EVAL_DATASETS 로 평가 집합을 바꾼다 (쉼표 구분, 예: "FridgeApple,FridgeOnion").
# 기본값은 기존 run 들이 쓰던 집합 그대로라 과거 결과와 계속 비교할 수 있다.
#
# 명시 지정이 중요한 이유: collect_windows 는 DATA_ROOT 에 존재하는 데이터셋만 훑고
# 없으면 조용히 건너뛴다. 그래서 나중에 새 데이터셋이 디스크에 추가되면 같은 명령이
# 예전보다 넓은 집합을 평가하게 되고, 과거 수치와 비교 불가능해진다(실제로 FridgeApple
# 이 추가된 뒤 grapes 전용 run 들과 평가 집합이 어긋난 적이 있다).
_env_dataset_names = os.environ.get("G1_EVAL_DATASETS", "").strip()
_DEFAULT_DATASETS = _build_datasets(
    [n.strip() for n in _env_dataset_names.split(",") if n.strip()]
    if _env_dataset_names
    else _DEFAULT_DATASET_NAMES
)

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

# 쿼터니언 구간. 45D 액션 레이아웃에는 회전이 없으므로 기본은 None 이고,
# --target state 에서 81D 레이아웃으로 바뀔 때만 설정된다. None 이면 아래
# quat/* 지표는 아예 생성되지 않으므로 기존 출력과 동일하다.
QUAT_SLICE = None

# 손·head 를 뺀 집계 차원과 손 레이아웃. 기본은 45D 액션 레이아웃이고,
# --target state 에서 81D 로 교체된다. 둘 다 추가 지표만 만들며 기존 키는
# 건드리지 않는다.
from state_layout import ACTION45_CLEAN_DIMS as _A45_CLEAN, HAND45 as _HAND45  # noqa: E402
CLEAN_DIMS = _A45_CLEAN
HAND_LAYOUT = _HAND45
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


def collect_windows(stride: int, max_windows_per_ep: int, target_split: str = "both",
                    target: str = "action", state_lead: int = 2):
    """모든 에피소드에서 (윈도 메타, GT, persistence 예측, 원시 state) 수집.

    target="action" (기본) 이면 기존과 동일하게 45D 액션을 GT 로 쓴다.
    target="state" 이면 state 를 타깃으로 학습한 체크포인트를 재려고 GT 를
    state[t+state_lead : t+state_lead+H] (81D) 로, persistence 를 state[t] 유지로
    바꾼다. 모델이 t 시점을 보므로 "마지막으로 관측한 상태를 유지"가 이 경우의
    zero-motion 기준선이다.
    """
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
                # target="state" 일 때만 GT/persistence 를 state 기준으로 교체한다.
                # 기본값에서는 이 블록에 들어오지 않으므로 위 결과가 그대로 쓰인다.
                if target == "state":
                    s0 = t + state_lead
                    gt_state = st_model[s0:s0 + H]
                    if len(gt_state) < H:  # 에피소드 끝: 마지막 프레임으로 패딩
                        pad = np.tile(st_model[-1], (H - len(gt_state), 1))
                        gt_state = np.concatenate([gt_state, pad], axis=0)
                    windows[-1]["gt"] = gt_state                              # (H,81)
                    windows[-1]["persist"] = np.tile(st_model[t], (H, 1))     # (H,81)
        print(f"[collect] {ds_name} ({split_type.upper()}): {len(eps)} eps -> 누적 {len(windows)} windows")
    return windows


def eval_model(ckpt_path: str, windows, batch_size: int, tag: str, send_state: bool = True,
               device: str = "cuda", logits_sink: list | None = None):
    """PolicyServerWrapper로 예측 → 비정규화 액션 (N,H,45).

    logits_sink 를 주면 손 분류 로짓을 배치마다 모아 담는다. 래퍼는 임계값 0 에서
    로짓을 손 명령으로 바꿔 81D 안에 써 넣고 로짓 자체는 버리는데, 그러면 임계값을
    바꿔볼 때마다 모델을 다시 돌려야 한다. 손 헤드가 없는 체크포인트에서는 아무것도
    담기지 않는다. 기본값 None 이면 기존 동작과 동일하다.
    """
    import random
    import torch
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    def _seed_everything(seed: int):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    _seed_everything(EVAL_SEED)
    wrapper = PolicyServerWrapper(ckpt_path=ckpt_path, device=device, use_bf16=True)
    _seed_everything(EVAL_SEED)
    # 출력 폭은 첫 배치를 받고 나서 정한다. 45D 액션 체크포인트든 81D state
    # 체크포인트든 같은 코드로 받기 위해서다. 폭을 여기서 45 로 고정해두면
    # state 체크포인트에서 윈도 수집을 마친 뒤에야 broadcast 오류로 터진다.
    preds = None
    for s in range(0, len(windows), batch_size):
        chunk = list(range(s, min(s + batch_size, len(windows))))
        examples = [dict(image=windows[i]["images"], lang=windows[i]["lang"]) for i in chunk]
        if send_state:
            for example, i in zip(examples, chunk):
                example["state"] = windows[i]["state"][None, :]
        # 손 헤드가 있는 체크포인트는 현재 손 자세를 입력으로 요구한다. 손 헤드가
        # 없는 모델은 이 키를 무시하므로 항상 넣어도 안전하다. 기준은 state[t] —
        # 수집 버그로 86% 에피소드에서 손 state 가 명령의 에코라 action[t-1] 과
        # 한 프레임 차이이고, 손 클래스는 1.8% 의 스텝에서만 바뀌므로 무해하다.
        if HAND_LAYOUT is not None and "state" in windows[chunk[0]]:
            from state_layout import hand_engaged, hand_thumb_flexed
            from state_layout import HAND81 as _H81
            for example, i in zip(examples, chunk):
                st = windows[i]["state"][None, :]
                rows = []
                for hand in ("L_hand", "R_hand"):
                    on = bool(hand_engaged(st, hand, _H81)[0])
                    md = float(np.rint(abs(st[0, _H81[hand]["mode"]])))
                    fx = bool(hand_thumb_flexed(st, hand, _H81)[0])
                    rows.append([float(on), md, float(fx)])
                example["hand_prev"] = np.asarray(rows, dtype=np.float32)
        out = wrapper.predict_action(examples=examples, state_is_normalized=False)
        acts = np.asarray(out["actions"], dtype=np.float32)  # (B, H, D)
        if logits_sink is not None and out.get("hand_logits") is not None:
            logits_sink.append(np.asarray(out["hand_logits"], dtype=np.float32))
        if preds is None:
            preds = np.zeros((len(windows), H, acts.shape[-1]), dtype=np.float32)
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

    # 회전 전용 지표. part/base_quat 은 성분별 MSE 라서 쿼터니언에는 쓸 수 없다:
    # q 와 -q 는 같은 회전인데 성분별 거리는 2(1-<q1,q2>) 에서 2(1+<q1,q2>) 로 튄다.
    # 6D 로 학습한 모델은 복원 시 정규 부호를 택하고 저장된 GT 의 부호는 임의라
    # 일부 프레임에서 반대 반구가 된다. 측정값: 3200 표본 중 2.5% 가 뒤집혔고
    # 그것이 part/base_quat 의 96.2% 를 만들었다. 그 지표로는 기준선의 17.2배로
    # 보이던 모델이 부호를 맞추면 0.66배, 즉 기준선보다 낫다.
    #
    # 아래 키는 전부 추가분이다. 기존 키는 손대지 않으므로 과거 결과와 계속
    # 비교할 수 있다.
    # 손·head 를 제외한 집계와 손 전이 검출 지표. 둘 다 추가분이며 CLEAN_DIMS 가
    # 설정된 --target state 에서만 생성되므로 기존 출력은 그대로다.
    #
    # 왜 필요한가: 손의 절대 오차가 팔의 20배라 43차원 평균에서 손이 지표를
    # 지배한다. 140k 평가에서 팔·다리가 13~18% 개선됐는데 전체 val 은 1% 만
    # 움직였고, 그 차이는 전부 손·head 가 정체한 탓이었다. 그리고 손은 99%
    # 프레임에서 "그대로"가 정답이라 집계 오차로는 전이 검출 능력이 보이지
    # 않는다 — 완벽한 검출기와 아무것도 안 하는 모델의 집계 차이가 작다.
    if CLEAN_DIMS is not None:
        res["overall_clean"] = float(err2[..., CLEAN_DIMS].mean())
        res["overall_mae_clean"] = float(err1[..., CLEAN_DIMS].mean())
        for sp in splits:
            sp_mask = np.array([w["split_type"] == sp for w in windows])
            res[f"overall_clean_{sp}"] = float(err2[sp_mask][..., CLEAN_DIMS].mean())

    if HAND_LAYOUT is not None:
        from state_layout import hand_engaged, hand_thumb_flexed

        # 기준 상태는 persistence 예측 그 자체다 — target=state 면 state[t] 유지,
        # target=action 이면 action[t-1] 유지. 둘 다 "변화 없음"의 올바른 기준이라
        # 레이아웃과 무관하게 같은 코드로 처리된다.
        base = np.stack([w["persist"] for w in windows])          # (N, H, D)
        for hand in HAND_LAYOUT:
            g_on = hand_engaged(gt, hand, HAND_LAYOUT)            # (N, H)
            p_on = hand_engaged(pred, hand, HAND_LAYOUT)
            c_on = hand_engaged(base, hand, HAND_LAYOUT)          # (N, H), 전 스텝 동일
            # "전이" = 그 스텝의 정답이 현재 상태와 다른 경우.
            gt_tr = g_on != c_on
            pr_tr = p_on != c_on
            tp = int((gt_tr & pr_tr).sum())
            fp = int((~gt_tr & pr_tr).sum())
            fn = int((gt_tr & ~pr_tr).sum())
            pre = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            k = f"hand/{hand}"
            res[f"{k}/transition_rate"] = float(gt_tr.mean())
            res[f"{k}/precision"] = float(pre)
            res[f"{k}/recall"] = float(rec)
            res[f"{k}/f1"] = float(2 * pre * rec / (pre + rec)) if (tp + fp) and (tp + fn) and (pre + rec) > 0 else float("nan")
            res[f"{k}/pred_transition_rate"] = float(pr_tr.mean())
            res[f"{k}/engaged_accuracy"] = float((g_on == p_on).mean())
            # 엄지 굽힘도 같은 방식으로. on 구간에서만 의미가 있다.
            g_tf, p_tf = hand_thumb_flexed(gt, hand, HAND_LAYOUT), hand_thumb_flexed(pred, hand, HAND_LAYOUT)
            m = g_on
            res[f"{k}/thumb_flex_accuracy_on"] = (
                float((g_tf[m] == p_tf[m]).mean()) if m.any() else float("nan")
            )
            for sp in splits:
                sp_mask = np.array([w["split_type"] == sp for w in windows])
                gt_s, pr_s = gt_tr[sp_mask], pr_tr[sp_mask]
                tp_s = int((gt_s & pr_s).sum()); fp_s = int((~gt_s & pr_s).sum()); fn_s = int((gt_s & ~pr_s).sum())
                res[f"{k}/precision_{sp}"] = float(tp_s / (tp_s + fp_s)) if (tp_s + fp_s) else float("nan")
                res[f"{k}/recall_{sp}"] = float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) else float("nan")

    if QUAT_SLICE is not None:
        qs, qe = QUAT_SLICE
        qp = pred[..., qs:qe].astype(np.float64)
        qg = gt[..., qs:qe].astype(np.float64)
        np_ = np.linalg.norm(qp, axis=-1, keepdims=True)
        ng = np.linalg.norm(qg, axis=-1, keepdims=True)
        qpn = qp / np.clip(np_, 1e-9, None)
        qgn = qg / np.clip(ng, 1e-9, None)
        dot = (qpn * qgn).sum(axis=-1)
        # 부호 정렬 후 성분별 MSE: 이중 덮개만 제거하고 나머지는 그대로다.
        sgn = np.where(dot < 0.0, -1.0, 1.0)[..., None]
        res["quat/mse_signfix"] = float(((sgn * qp - qg) ** 2).mean())
        # 측지 각도(도). 물리적으로 해석되는 유일한 회전 오차.
        deg = np.degrees(2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))
        res["quat/geodesic_deg_mean"] = float(deg.mean())
        res["quat/geodesic_deg_median"] = float(np.median(deg))
        res["quat/geodesic_deg_p90"] = float(np.percentile(deg, 90))
        res["quat/geodesic_deg_p99"] = float(np.percentile(deg, 99))
        res["quat/sign_flip_frac"] = float((dot < 0.0).mean())
        for sp in splits:
            sp_mask = np.array([w["split_type"] == sp for w in windows])
            res[f"quat/geodesic_deg_mean_{sp}"] = float(deg[sp_mask].mean())
    return res


def _dump_hand_logits(path_base: str, label: str, sink: list, windows) -> None:
    """로짓과 정답/현재 on/off 를 함께 저장해 임계값 스윕을 모델 없이 돌릴 수 있게 한다.

    전이 정의가 "정답이 현재 상태와 다름"이라 정답만으로는 부족하고 persistence
    기준도 같이 있어야 한다. 스플릿도 담는다 — 임계값은 val 에서 골라야 한다.
    """
    from state_layout import hand_engaged

    logits = np.concatenate(sink, axis=0)                      # (N, H, 6*손개수)
    if logits.shape[0] != len(windows):
        print(f"[경고] 로짓 {logits.shape[0]}개가 윈도 {len(windows)}개와 다르다 — 저장 생략")
        return
    gt = np.stack([w["gt"] for w in windows])
    base = np.stack([w["persist"] for w in windows])
    hands = list(HAND_LAYOUT) if HAND_LAYOUT else []
    data = {
        "logits": logits,
        "split": np.array([w["split_type"] for w in windows]),
        "tag": np.array([w["display_tag"] for w in windows]),
        "hands": np.array(hands),
    }
    for hand in hands:
        data[f"gt_on/{hand}"] = hand_engaged(gt, hand, HAND_LAYOUT)
        data[f"cur_on/{hand}"] = hand_engaged(base, hand, HAND_LAYOUT)
    out = f"{path_base}.{label}.npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"[dump] 손 로짓 -> {out}  logits{logits.shape}, 손 {hands}")


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
    ap.add_argument("--target", choices=["action", "state"], default="action",
                    help="action (기본): 45D 액션 GT. state: state 를 타깃으로 학습한 "
                         "체크포인트 평가용으로 81D state GT 와 state 기준 persistence 사용")
    ap.add_argument("--state_lead", type=int, default=2,
                    help="--target state 일 때 GT 의 선행 프레임 수 "
                         "(학습 설정의 state_lead_frames 와 맞출 것)")
    ap.add_argument("--dump_hand_logits", default=None,
                    help="손 분류 로짓과 정답/현재 on/off 를 '<경로>.<label>.npz' 로 저장한다. "
                         "sweep_hand_threshold.py 가 이 파일로 임계값을 모델 없이 스윕한다. "
                         "손 헤드가 없는 체크포인트에서는 아무 일도 하지 않는다.")
    ap.add_argument("--device", default="cuda",
                    help="추론 device (기본 cuda). 같은 GPU 에서 학습이 돌고 있어 "
                         "메모리가 없을 때 cpu 로 돌릴 수 있다 (매우 느림)")
    args = ap.parse_args()

    # state 모드에서만 집계 레이아웃을 81D 로 교체한다. summarize() 는 이 전역들을
    # 호출 시점에 읽으므로 함수 자체는 손대지 않는다.
    if args.target == "state":
        from state_layout import STATE81_PARTS, STATE_CONT_DIMS, STATE_MODE_DIMS
        globals()["PARTS"] = STATE81_PARTS
        globals()["CONT_DIMS"] = STATE_CONT_DIMS
        globals()["MODE_DIMS"] = STATE_MODE_DIMS
        from state_layout import STATE_QUAT_SLICE, STATE81_CLEAN_DIMS, HAND81
        globals()["QUAT_SLICE"] = STATE_QUAT_SLICE
        globals()["CLEAN_DIMS"] = STATE81_CLEAN_DIMS
        globals()["HAND_LAYOUT"] = HAND81
        print(f"[target] state 81D, lead {args.state_lead} 프레임 — "
              f"감독 차원만 집계 (연속 {len(STATE_CONT_DIMS)} + 모드 {len(STATE_MODE_DIMS)})")

    windows = collect_windows(args.stride, args.max_windows_per_ep, target_split=args.split,
                              target=args.target, state_lead=args.state_lead)
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
        sink = [] if args.dump_hand_logits else None
        results[label] = summarize(
            eval_model(ck, windows, args.batch_size, label, send_state=not args.no_send_state,
                       device=args.device, logits_sink=sink), windows)
        if sink:
            _dump_hand_logits(args.dump_hand_logits, label, sink, windows)
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
