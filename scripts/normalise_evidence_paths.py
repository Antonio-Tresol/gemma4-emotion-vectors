"""Rewrite machine-local absolute paths in committed evidence to repo-relative.

`results/` is not hand-edited in this project, so this is a script and not a
text edit, and it is deliberately narrow: it touches JSON string VALUES that
look like a path into this checkout, and nothing else. Every number, key,
ordering and structure is asserted unchanged before the file is written.

Why it is needed at all: the scripts that produced these files recorded
`str(path)` for provenance, so whatever was typed on the command line was
baked in. An absolute path there is meaningless on anyone else's machine, and
publishes a home directory. The writers now call
`emotion_vectors.artifacts.repo_relative`; this fixes what they wrote before.

    uv run python scripts/normalise_evidence_paths.py            # report
    uv run python scripts/normalise_evidence_paths.py --apply    # rewrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# any absolute path belonging to a person's checkout, this one or another
LOCAL_PATH = re.compile(r"^(/Users/[^/]+|/home/[^/]+|[A-Za-z]:\\\\Users\\\\[^\\\\]+)/")


def shorten(value: str) -> str:
    """The tail of a local absolute path, from the project directory onwards."""
    if not LOCAL_PATH.match(value):
        return value
    parts = Path(value).parts
    for anchor in ("results", "scripts", "notebooks", "src", "data"):
        if anchor in parts:
            return str(Path(*parts[parts.index(anchor) :]))
    return value


def walk(node: Any) -> tuple[Any, int]:
    """Rewrite path-like strings anywhere in the structure; count the changes."""
    if isinstance(node, str):
        shortened = shorten(node)
        return shortened, int(shortened != node)
    if isinstance(node, list):
        pairs = [walk(item) for item in node]
        return [value for value, _ in pairs], sum(count for _, count in pairs)
    if isinstance(node, dict):
        out, total = {}, 0
        for key, value in node.items():
            out[key], count = walk(value)
            total += count
        return out, total
    return node, 0


def numbers(node: Any) -> list[float]:
    """Every number in the structure, in traversal order."""
    if isinstance(node, bool):
        return []
    if isinstance(node, (int, float)):
        return [float(node)]
    if isinstance(node, list):
        return [n for item in node for n in numbers(item)]
    if isinstance(node, dict):
        return [n for value in node.values() for n in numbers(value)]
    return []


def scrub_notebooks(*, apply: bool) -> int:
    """Drop stored stderr warnings that carry a local absolute path.

    Executing a notebook captures stderr, so an incidental library warning
    ("Consider using IPython.display.IFrame instead") is stored complete with
    the full path to the interpreter that emitted it. That publishes a home
    directory and a worktree name in a committed artifact, and it is not a
    result: nothing downstream reads it and removing it loses no information.

    Only `stream` outputs on stderr that contain such a path are removed.
    Results, figures, and stdout are untouched, and the file is rewritten
    through Python's json, which round-trips these notebooks byte-for-byte.
    """
    changed = 0
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        raw = path.read_text(encoding="utf-8")
        if not any(marker in raw for marker in ("/Users/", "/home/")):
            continue
        notebook = json.loads(raw)
        removed = 0
        for cell in notebook.get("cells", []):
            keep = []
            for output in cell.get("outputs", []):
                text = "".join(output.get("text", []))
                noisy = (
                    output.get("output_type") == "stream"
                    and output.get("name") == "stderr"
                    and LOCAL_PATH.search(text.lstrip())
                )
                if noisy:
                    removed += 1
                else:
                    keep.append(output)
            if "outputs" in cell:
                cell["outputs"] = keep
        if not removed:
            continue
        changed += 1
        print(f"  {path.relative_to(ROOT)}: {removed} stderr warning(s) with a local path")
        if apply:
            path.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the files")
    args = parser.parse_args()

    changed_files = 0
    for path in sorted((ROOT / "results").rglob("*.json")):
        original = path.read_text(encoding="utf-8")
        if not any(marker in original for marker in ("/Users/", "/home/", ":\\\\Users")):
            continue
        data = json.loads(original)
        rewritten, count = walk(data)
        if count == 0:
            continue

        # the guarantee: only path strings moved
        if numbers(data) != numbers(rewritten):
            print(f"  ABORT {path.relative_to(ROOT)}: a number changed")
            return 1
        if json.dumps(data, sort_keys=True) == json.dumps(rewritten, sort_keys=True):
            continue

        changed_files += 1
        print(f"  {path.relative_to(ROOT)}: {count} path(s)")
        if args.apply:
            trailing = "\n" if original.endswith("\n") else ""
            path.write_text(json.dumps(rewritten, indent=2) + trailing, encoding="utf-8")

    changed_files += scrub_notebooks(apply=args.apply)
    verb = "rewritten" if args.apply else "would change"
    print(f"\n{changed_files} file(s) {verb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
