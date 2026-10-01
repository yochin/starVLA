"""Assemble a deployable bundle from a training run and push it to a HF repo.

A bare checkpoint cannot be loaded: `share_tools.read_mode_config` asserts on
`config.yaml` and `dataset_statistics.json` sitting **two directories up** from the
`.pt`. So this collects those alongside the weights and preserves the layout.

The checkpoint is pinned by step rather than taken from `final_model/`, because
`final_model/pytorch_model.pt` is overwritten whenever a run is resumed — the
state6dres run went through it twice (80k, then 160k). Pass --step to pick, or
--final to accept the current final_model and record which step it is.

Authenticate once, in your own terminal so the token stays out of any transcript:

    hf auth login          # a fine-grained token with write access to the repo

Then, for example:

    python deployment/upload/push_run_to_hf.py \\
        --run starvla_qwenoft_g1_fridge5_state6dreshand --step 80000 \\
        --repo_id yochin/DF --subdir 04_state6dreshand_80k

--dry_run reports what would be sent and touches nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO_ROOT / "results" / "Checkpoints"
REQUIRED_META = ("config.yaml", "dataset_statistics.json")
OPTIONAL_META = ("config.full.yaml",)


def _resolve_checkpoint(run_dir: Path, step: int | None, use_final: bool) -> tuple[Path, str]:
    """Return (checkpoint path, a label describing which step it is)."""
    if use_final:
        p = run_dir / "final_model" / "pytorch_model.pt"
        if not p.is_file():
            raise FileNotFoundError(f"no final_model checkpoint at {p}")
        return p, "final"
    ckpts = run_dir / "checkpoints"
    found = {}
    if ckpts.is_dir():
        for f in ckpts.iterdir():
            m = re.match(r"steps_(\d+)_pytorch_model\.pt$", f.name)
            if m:
                found[int(m.group(1))] = f
    if not found:
        raise FileNotFoundError(
            f"no steps_<N>_pytorch_model.pt under {ckpts}. Pass --final to use "
            "final_model/ instead, but note it is overwritten on resume."
        )
    if step is None:
        step = max(found)
        print(f"[info] --step not given; taking the latest, {step}")
    if step not in found:
        raise FileNotFoundError(
            f"step {step} not found. Available: {sorted(found)}"
        )
    return found[step], f"step {step}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory name under results/Checkpoints")
    ap.add_argument("--repo_id", required=True, help="e.g. yochin/DF")
    ap.add_argument("--subdir", default=None, help="Path inside the repo (default: the run name)")
    ap.add_argument("--step", type=int, default=None, help="Checkpoint step to pin")
    ap.add_argument("--final", action="store_true", help="Use final_model/ instead of a pinned step")
    ap.add_argument("--public", action="store_true", help="Create the repo public (default private)")
    ap.add_argument(
        "--include",
        action="append",
        default=[],
        help="Extra glob(s) relative to the run dir to ship, e.g. '*.json'. Repeatable.",
    )
    ap.add_argument(
        "--note",
        default=None,
        help="Markdown file to ship as DEPLOY.md — the client contract for this model.",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    run_dir = CKPT_ROOT / args.run
    if not run_dir.is_dir():
        print(f"[error] no such run: {run_dir}")
        return 1

    try:
        ckpt, label = _resolve_checkpoint(run_dir, args.step, args.final)
    except (FileNotFoundError, OSError) as exc:
        print(f"[error] {exc}")
        return 1

    missing = [m for m in REQUIRED_META if not (run_dir / m).is_file()]
    if missing:
        print(f"[error] {run_dir} is missing {missing} — the loader asserts on these.")
        return 1

    subdir = args.subdir or args.run
    size = ckpt.stat().st_size
    print(f"run        : {run_dir}")
    print(f"checkpoint : {ckpt.name}  ({label}, {size / 2**30:.2f} GiB)")
    print(f"metadata   : {[m for m in REQUIRED_META + OPTIONAL_META if (run_dir / m).is_file()]}")
    print(f"target     : {args.repo_id}/{subdir}  (private={not args.public})")

    # Mirror the on-disk layout the loader expects: the .pt two levels below the
    # metadata. Hard-link the weights so staging costs no extra space.
    rel = "checkpoints" if not args.final else "final_model"

    # 필수·선택 메타는 아래에서 따로 복사하므로 글롭에서 빼 둔다. 안 그러면
    # dataset_statistics.json 처럼 두 번 잡혀 목록이 헷갈린다.
    already = set(REQUIRED_META + OPTIONAL_META)
    extras, seen = [], set()
    for pat in args.include:
        for q in sorted(run_dir.glob(pat)):
            if q.is_file() and q.name not in already and q.name not in seen:
                extras.append(q)
                seen.add(q.name)
    if extras:
        print(f"extras     : {[q.name for q in extras]}")
    note = Path(args.note) if args.note else None
    if note is not None and not note.is_file():
        print(f"[error] --note file not found: {note}")
        return 1
    if note is not None:
        print(f"note       : {note} -> DEPLOY.md")

    if args.dry_run:
        print(f"\nwould upload:\n   {subdir}/{rel}/{ckpt.name}")
        for m in REQUIRED_META + OPTIONAL_META:
            if (run_dir / m).is_file():
                print(f"   {subdir}/{m}")
        for q in extras:
            print(f"   {subdir}/{q.name}")
        if note is not None:
            print(f"   {subdir}/DEPLOY.md")
        print("\n[dry run] nothing uploaded.")
        return 0

    from huggingface_hub import HfApi, create_repo, whoami

    try:
        who = whoami()
    except Exception as exc:  # noqa: BLE001 — any auth failure reads the same
        print(f"\n[error] not authenticated ({type(exc).__name__})")
        print("Run 'hf auth login' and paste a fine-grained write token.")
        return 1
    print(f"\nauthenticated as: {who.get('name')}")

    staging = Path(tempfile.mkdtemp(prefix="hf_bundle_", dir=str(ckpt.parent.parent)))
    try:
        dest = staging / subdir
        (dest / rel).mkdir(parents=True)
        try:
            (dest / rel / ckpt.name).hardlink_to(ckpt)
        except OSError:
            shutil.copy2(ckpt, dest / rel / ckpt.name)
        for m in REQUIRED_META + OPTIONAL_META:
            if (run_dir / m).is_file():
                shutil.copy2(run_dir / m, dest / m)
        for q in extras:
            shutil.copy2(q, dest / q.name)
        if note is not None:
            shutil.copy2(note, dest / "DEPLOY.md")
        # A note so whoever downloads it knows which step this is and what it needs.
        (dest / "PROVENANCE.json").write_text(
            json.dumps(
                {
                    "run": args.run,
                    "checkpoint": ckpt.name,
                    "which": label,
                    "loader_requires": list(REQUIRED_META),
                    "layout_note": (
                        "config.yaml and dataset_statistics.json must stay two "
                        "directories above the .pt file."
                    ),
                },
                indent=1,
            )
        )

        create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
        HfApi().upload_large_folder(
            repo_id=args.repo_id, folder_path=str(staging), repo_type="model"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\ndone: https://huggingface.co/{args.repo_id}/tree/main/{subdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
