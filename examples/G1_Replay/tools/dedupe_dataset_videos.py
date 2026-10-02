"""두 데이터셋 트리에서 내용이 같은 파일을 하드링크로 합쳐 디스크를 회수한다.

fridge_picktake_allbottles 의 passthrough / nopassthrough 는 같은 녹화를 두 번
export 한 것이다. 코어 텐서(observation.state 181, action 154)가 바이트 단위로
동일하고 영상 4종도 md5 가 같다. 차이는 passthrough 쪽에 raw 텔레메트리 214
컬럼이 더 있다는 것뿐이다. 영상이 각 150GB 라 한쪽이 순수 중복이다.

안전장치:
  * 기본은 dry-run 이다. --apply 를 줘야 실제로 링크한다.
  * 크기가 같은 후보만 md5 를 재고, **전체 해시가 일치할 때만** 링크한다.
  * 이미 같은 inode 면 건너뛴다(두 번 돌려도 안전).
  * 링크는 원자적으로 교체한다(같은 디렉터리에 임시 이름 → os.replace).
    교체 전 임시 링크 생성이 실패하면 원본은 그대로 남는다.

하드링크는 내용을 공유하므로 한쪽을 **제자리에서 수정하면 양쪽이 바뀐다**.
녹화 데이터는 불변이라 괜찮지만, 편집할 계획이 있다면 쓰지 말 것.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path


def _md5(p: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _scan(root: Path, suffixes: tuple[str, ...]) -> dict[int, list[Path]]:
    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in suffixes:
            by_size[p.stat().st_size].append(p)
    return by_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="기준 트리 (이 쪽 파일을 남긴다)")
    ap.add_argument("--b", required=True, help="중복 트리 (이 쪽을 링크로 교체한다)")
    ap.add_argument("--suffix", default=".mp4", help="대상 확장자 (쉼표 구분, 기본 .mp4)")
    ap.add_argument("--apply", action="store_true", help="실제로 링크한다 (기본은 dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="처리할 최대 쌍 수 (0=제한 없음)")
    args = ap.parse_args()

    A, B = Path(args.a), Path(args.b)
    for r in (A, B):
        if not r.is_dir():
            print(f"[오류] 디렉터리가 없다: {r}")
            return 1
    sfx = tuple(s if s.startswith(".") else "." + s for s in args.suffix.split(","))

    print(f"기준 : {A}")
    print(f"중복 : {B}")
    print(f"대상 : {sfx}   모드: {'적용' if args.apply else 'dry-run'}")
    a_by_size = _scan(A, sfx)
    b_by_size = _scan(B, sfx)
    print(f"파일 수: A {sum(len(v) for v in a_by_size.values())}, "
          f"B {sum(len(v) for v in b_by_size.values())}")

    shared = sorted(set(a_by_size) & set(b_by_size))
    print(f"같은 크기가 양쪽에 있는 크기 값: {len(shared)}개")

    # 크기별로 A 쪽 해시를 만들고 B 를 맞춘다. 크기가 같은 후보만 읽으므로
    # 전체 트리를 해시하지 않는다.
    linked = already = mismatch = 0
    saved = 0
    pairs = 0
    for size in shared:
        a_files, b_files = a_by_size[size], b_by_size[size]
        a_hash: dict[str, Path] = {}
        for p in a_files:
            a_hash.setdefault(_md5(p), p)
        for q in b_files:
            if args.limit and pairs >= args.limit:
                break
            h = _md5(q)
            src = a_hash.get(h)
            if src is None:
                mismatch += 1
                continue
            pairs += 1
            if os.path.samefile(src, q):
                already += 1
                continue
            saved += size
            if not args.apply:
                continue
            tmp = q.with_name(q.name + ".dedupe_tmp")
            try:
                if tmp.exists():
                    tmp.unlink()
                os.link(src, tmp)
                os.replace(tmp, q)
                linked += 1
            except OSError as exc:
                print(f"   [실패] {q}: {exc}")
                if tmp.exists():
                    tmp.unlink()
        if args.limit and pairs >= args.limit:
            break

    print(f"\n내용 일치 쌍      : {pairs}")
    print(f"이미 같은 inode   : {already}")
    print(f"크기만 같고 내용 다름: {mismatch}")
    print(f"회수 가능/회수량  : {saved / 2**30:.1f} GiB")
    if args.apply:
        print(f"링크 완료         : {linked}개")
    else:
        print("\n[dry-run] 아무것도 바꾸지 않았다. --apply 로 실행한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
