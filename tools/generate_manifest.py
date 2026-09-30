#!/usr/bin/env python3
"""Generate the release SHA-256 manifest reproducibly on Linux or macOS."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"

SKIP_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "IMG_CI",
    "IMG_TEST",
    "reproduced_results",
}
SKIP_NAMES = {"MANIFEST.sha256", ".DS_Store"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path.name in SKIP_NAMES or path.suffix in {".pyc", ".pyo"}:
        return False
    if any(part in SKIP_DIRECTORIES or part.endswith(".egg-info") for part in relative.parts):
        return False
    return path.is_file()


def main() -> None:
    files = sorted(
        (path for path in ROOT.rglob("*") if included(path)),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    lines = [
        f"{sha256(path)}  ./{path.relative_to(ROOT).as_posix()}"
        for path in files
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(lines)} entries.")


if __name__ == "__main__":
    main()
