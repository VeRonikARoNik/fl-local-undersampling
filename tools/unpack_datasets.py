#!/usr/bin/env python3
"""Unpack the data sets in datasets/*.zip into datasets_original/, the folder the code expects.

Usage, from the root of the repository:

    python tools/unpack_datasets.py            # unpack every archive that is not unpacked yet
    python tools/unpack_datasets.py --force    # unpack again and overwrite

Each archive holds one folder with the name that the split scripts expect (for example car+evaluation/).
UNSW-NB15 is not redistributed here; the script prints where to download it (see datasets/README.md).
Works the same on Windows, macOS and Linux.
"""
import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNSW_URL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
UNSW_FILES = ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")


def safe_extract(zf: zipfile.ZipFile, target: Path) -> None:
    """Extract an archive, refusing members that would land outside the target folder."""
    target = target.resolve()
    for member in zf.infolist():
        if not (target / member.filename).resolve().is_relative_to(target):
            raise RuntimeError(f"unsafe path in archive: {member.filename}")
    zf.extractall(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unpack datasets/*.zip into datasets_original/.")
    parser.add_argument("--source", type=Path, default=ROOT / "datasets", help="folder with the zip archives")
    parser.add_argument("--target", type=Path, default=ROOT / "datasets_original", help="folder to unpack into")
    parser.add_argument("--force", action="store_true", help="unpack again even if the folder already exists")
    args = parser.parse_args()

    archives = sorted(args.source.glob("*.zip"))
    if not archives:
        print(f"No zip archives found in {args.source}", file=sys.stderr)
        return 1

    args.target.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            folders = sorted({name.split("/")[0] for name in zf.namelist()})
            if not args.force and all((args.target / f).exists() for f in folders):
                print(f"skipped   {archive.name} (already unpacked)")
                continue
            safe_extract(zf, args.target)
            print(f"unpacked  {archive.name} -> {args.target.name}/{', '.join(folders)}")

    unsw = args.target / "unsw-nb15"
    if not all((unsw / f).exists() for f in UNSW_FILES):
        print()
        print("UNSW-NB15 is not included, because its terms do not cover redistribution.")
        print(f"Download {UNSW_FILES[0]} and {UNSW_FILES[1]} from")
        print(f"  {UNSW_URL}")
        print(f"into {unsw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
