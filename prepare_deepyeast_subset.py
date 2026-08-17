#!/usr/bin/env python3
"""Download and organize a 5% stratified subset of the DeepYeast main dataset."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

BASE_URL = "http://kodu.ut.ee/~leopoldp/2016_DeepYeast"
CACHE_DIR = Path.home() / ".deepyeast" / "cache"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "deepyeast_5pct"

MANIFEST_FILES = {
    "HOwt_doc.txt": (
        f"{BASE_URL}/code/image_prep/data/HOwt_doc.txt",
        "33b7780020972e2da4f884c6b5a63b25",
    ),
    "HOwt_train.txt": (
        f"{BASE_URL}/code/image_prep/data/HOwt_train.txt",
        "b71eb4ff50f955adfa72048c6d8c0233",
    ),
    "HOwt_val.txt": (
        f"{BASE_URL}/code/image_prep/data/HOwt_val.txt",
        "2ac1d1874b89d6a1ad3d948d36c1e229",
    ),
    "HOwt_test.txt": (
        f"{BASE_URL}/code/image_prep/data/HOwt_test.txt",
        "c7958faa20232ff52fb196e754645bd1",
    ),
}

TAR_FILE = "main.tar.gz"
TAR_URL = f"{BASE_URL}/data/{TAR_FILE}"
# Official deepyeast dataset.py lists f313fc0b806894f9c82725fffe3b096b6, but the
# file currently served at kodu.ut.ee has this checksum.
TAR_MD5 = "f313fc0b8068941ab18ae65eb113afee"
TAR_EXPECTED_SIZE = 417335070

SPLITS = ("train", "val", "test")
SPLIT_MANIFEST = {
    "train": "HOwt_train.txt",
    "val": "HOwt_val.txt",
    "test": "HOwt_test.txt",
}


def md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        actual_md5 = md5_file(dest)
        size_ok = dest.stat().st_size == TAR_EXPECTED_SIZE if dest.name == TAR_FILE else True
        if expected_md5 is None or (actual_md5 == expected_md5 and size_ok):
            print(f"  cached: {dest.name}")
            return dest
        print(
            f"  hash mismatch, re-downloading: {dest.name} "
            f"(got {actual_md5}, expected {expected_md5})"
        )

    print(f"  downloading: {url}")
    tmp = dest.parent / f"{dest.name}.part"
    result = subprocess.run(
        ["curl", "-fL", "--progress-bar", "-o", str(tmp), url],
        check=False,
    )
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url} (curl exit {result.returncode})")
    print()
    tmp.replace(dest)

    if expected_md5 and md5_file(dest) != expected_md5:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {dest.name}")
    return dest


def load_class_map(doc_path: Path) -> dict[int, str]:
    class_map: dict[int, str] = {}
    with doc_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("label"):
                continue
            name, idx = line.split(";")
            class_map[int(idx)] = name
    return class_map


def sanitize_class_name(name: str) -> str:
    return name.replace(" ", "_")


def parse_manifest(manifest_path: Path) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(" ", 1)
            entries.append((rel_path, int(label)))
    return entries


def stratified_sample(
    entries: list[tuple[str, int]], fraction: float, seed: int
) -> list[tuple[str, int]]:
    by_label: dict[int, list[str]] = defaultdict(list)
    for rel_path, label in entries:
        by_label[label].append(rel_path)

    rng = random.Random(seed)
    sampled: list[tuple[str, int]] = []
    for label in sorted(by_label):
        paths = by_label[label]
        k = max(1, round(len(paths) * fraction))
        chosen = rng.sample(paths, k)
        sampled.extend((p, label) for p in chosen)
    return sampled


def print_progress(done: int, total: int, prefix: str, start_time: float) -> None:
    pct = 100.0 * done / total if total else 100.0
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    bar_width = 30
    filled = int(bar_width * done / total) if total else bar_width
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(
        f"\r  {prefix} [{bar}] {done}/{total} ({pct:5.1f}%) "
        f"| {rate:5.1f} img/s | ETA {eta:5.0f}s"
    )
    sys.stdout.flush()


def extract_subset(
    tar_path: Path,
    samples: dict[str, list[tuple[str, int]]],
    class_map: dict[int, str],
    out_dir: Path,
) -> list[dict]:
    needed: dict[str, tuple[str, int]] = {}
    for split, items in samples.items():
        for rel_path, label in items:
            needed[rel_path] = (split, label)

    total = len(needed)
    rows: list[dict] = []
    found: set[str] = set()
    start = time.time()

    print(f"  Scanning tar and extracting {total} images (single pass)...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel_path = member.name
            if rel_path not in needed:
                continue

            split, label = needed[rel_path]
            class_name = class_map[label]
            class_folder = sanitize_class_name(class_name)
            dest_dir = out_dir / split / class_folder
            dest_dir.mkdir(parents=True, exist_ok=True)

            basename = Path(rel_path).name
            dest_path = dest_dir / basename

            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to extract: {member.name}")
            with dest_path.open("wb") as out_f:
                shutil.copyfileobj(extracted, out_f)

            rows.append(
                {
                    "split": split,
                    "filename": basename,
                    "relative_path": str(dest_path.relative_to(out_dir)),
                    "label_idx": label,
                    "label_name": class_name,
                    "class_folder": class_folder,
                    "source_path_in_tar": rel_path,
                }
            )
            found.add(rel_path)
            print_progress(len(found), total, "Extract", start)
            if len(found) == total:
                break

    print()
    missing = set(needed) - found
    if missing:
        example = next(iter(missing))
        raise FileNotFoundError(
            f"{len(missing)} images not found in tar (e.g. {example})"
        )
    return rows


def write_labels_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "split",
        "filename",
        "relative_path",
        "label_idx",
        "label_name",
        "class_folder",
        "source_path_in_tar",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_stats(rows: list[dict], class_map: dict[int, str]) -> dict:
    stats: dict = {
        "total_images": len(rows),
        "num_classes": len(class_map),
        "splits": {},
    }
    for split in SPLITS:
        split_rows = [r for r in rows if r["split"] == split]
        by_class: dict[str, int] = defaultdict(int)
        for r in split_rows:
            by_class[r["label_name"]] += 1
        stats["splits"][split] = {
            "total": len(split_rows),
            "num_classes_present": len(by_class),
            "per_class": dict(sorted(by_class.items())),
        }
    return stats


def verify_images(rows: list[dict], out_dir: Path, n_check: int = 3) -> list[dict]:
    rng = random.Random(0)
    check_rows = rng.sample(rows, min(n_check, len(rows)))
    checks = []
    for row in check_rows:
        path = out_dir / row["relative_path"]
        with Image.open(path) as img:
            arr = __import__("numpy").array(img)
            shape = arr.shape
        checks.append({"path": row["relative_path"], "shape": list(shape)})
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DeepYeast 5% subset")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for organized subset",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.05,
        help="Fraction of each class per split to sample (default: 0.05)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep downloaded tar.gz in cache (default: keep)",
    )
    parser.add_argument(
        "--delete-tar",
        action="store_true",
        help="Delete tar.gz from cache after extraction",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Download manifests")
    manifest_paths: dict[str, Path] = {}
    for name, (url, md5) in MANIFEST_FILES.items():
        manifest_paths[name] = download_file(url, cache_dir / name, md5)

    class_map = load_class_map(manifest_paths["HOwt_doc.txt"])
    class_map_json = {
        "index_to_name": {str(k): v for k, v in class_map.items()},
        "name_to_index": {v: k for k, v in class_map.items()},
        "folder_names": {
            v: sanitize_class_name(v) for v in class_map.values()
        },
    }

    print("Step 2: Stratified sampling")
    samples: dict[str, list[tuple[str, int]]] = {}
    for split in SPLITS:
        entries = parse_manifest(manifest_paths[SPLIT_MANIFEST[split]])
        samples[split] = stratified_sample(entries, args.fraction, args.seed)
        print(f"  {split}: {len(entries)} -> {len(samples[split])} images")

    print("Step 3: Download tar archive")
    tar_path = download_file(TAR_URL, cache_dir / TAR_FILE, TAR_MD5)

    print("Step 4: Extract subset and organize")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rows = extract_subset(tar_path, samples, class_map, out_dir)
    write_labels_csv(rows, out_dir / "labels.csv")

    with (out_dir / "class_map.json").open("w") as f:
        json.dump(class_map_json, f, indent=2)

    stats = build_stats(rows, class_map)
    image_checks = verify_images(rows, out_dir)
    stats["image_shape_checks"] = image_checks

    with (out_dir / "dataset_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    if args.delete_tar and not args.keep_tar:
        tar_path.unlink(missing_ok=True)
        print(f"Removed cache tar: {tar_path}")

    print("\nDone.")
    print(f"Output: {out_dir}")
    print(f"Total images: {stats['total_images']}")
    for split in SPLITS:
        s = stats["splits"][split]
        print(
            f"  {split}: {s['total']} images, "
            f"{s['num_classes_present']} classes"
        )
    print("Sample image shapes:", image_checks)


if __name__ == "__main__":
    main()
