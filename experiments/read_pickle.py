#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import pickle
import pprint
import sys
from pathlib import Path
from typing import Any


def ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read and inspect a pickle file.",
    )
    parser.add_argument("pickle_path", help="Path to the pickle file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many items to preview for containers. Default: 5.",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Print a specific item from a list/tuple.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the entire loaded object with pprint.",
    )
    return parser.parse_args()


def load_pickle(pickle_path: Path) -> Any:
    with pickle_path.open("rb") as f:
        return pickle.load(f)


def short_repr(value: Any, max_len: int = 500) -> str:
    text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def summarize_object(obj: Any, limit: int, indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}type: {type(obj).__module__}.{type(obj).__name__}")

    if isinstance(obj, dict):
        print(f"{prefix}len: {len(obj)}")
        print(f"{prefix}preview:")
        for idx, (key, value) in enumerate(obj.items()):
            if idx >= limit:
                print(f"{prefix}  ...")
                break
            print(f"{prefix}  [{idx}] key={short_repr(key)} value_type={type(value).__name__}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}len: {len(obj)}")
        print(f"{prefix}preview:")
        for idx, item in enumerate(obj[:limit]):
            print(f"{prefix}  [{idx}] {short_repr(item)}")
        if len(obj) > limit:
            print(f"{prefix}  ...")
    elif isinstance(obj, set):
        print(f"{prefix}len: {len(obj)}")
        print(f"{prefix}preview:")
        for idx, item in enumerate(list(obj)[:limit]):
            print(f"{prefix}  [{idx}] {short_repr(item)}")
        if len(obj) > limit:
            print(f"{prefix}  ...")
    elif hasattr(obj, "__dict__"):
        fields = vars(obj)
        print(f"{prefix}fields: {list(fields.keys())}")
        for key, value in fields.items():
            print(f"{prefix}  {key}: {short_repr(value)}")
    else:
        print(f"{prefix}value: {short_repr(obj)}")


def expand_for_print(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: expand_for_print(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {key: expand_for_print(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_for_print(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_for_print(item) for item in value)
    if isinstance(value, set):
        return {expand_for_print(item) for item in value}
    if hasattr(value, "__dict__"):
        return {
            key: expand_for_print(item)
            for key, item in vars(value).items()
        }
    return value


def main() -> int:
    ensure_repo_root_on_path()

    # Import project modules before unpickling so project-defined classes resolve.
    try:
        import experiments  # noqa: F401
        import experiments.generates.exp_bank  # noqa: F401
    except Exception:
        pass

    args = parse_args()
    pickle_path = Path(args.pickle_path).expanduser().resolve()

    if not pickle_path.exists():
        print(f"File not found: {pickle_path}", file=sys.stderr)
        return 1

    try:
        obj = load_pickle(pickle_path)
    except Exception as exc:
        print(f"Failed to load pickle: {exc}", file=sys.stderr)
        return 1

    print(f"path: {pickle_path}")
    summarize_object(obj, limit=args.limit)

    if args.index is not None:
        if not isinstance(obj, (list, tuple)):
            print("--index only works for list/tuple objects.", file=sys.stderr)
            return 1
        try:
            item = obj[args.index]
        except IndexError:
            print(f"Index out of range: {args.index}", file=sys.stderr)
            return 1
        print("\nselected_item:")
        pprint.pprint(expand_for_print(item), sort_dicts=False, width=120)

    if args.full:
        print("\nfull_object:")
        pprint.pprint(obj, sort_dicts=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
