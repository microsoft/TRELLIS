#!/usr/bin/env python3
"""Finalize exported vehicle teacher latents as a TRELLIS training dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SS_LATENT_MODEL = "ss_enc_conv3d_16l8_fp16"
SLAT_MODEL = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
SS_COLUMN = f"ss_latent_{SS_LATENT_MODEL}"
SLAT_COLUMN = f"latent_{SLAT_MODEL}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate teacher exports, build condition metadata, and split by car_id."
    )
    parser.add_argument("--manifest", required=True, help="The same filtered manifest used during export.")
    parser.add_argument("--teacher_dir", required=True, help="Output directory from export_vehicle_teacher_latents.py.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Dataset root. Defaults to teacher_dir; a different root links the teacher latent directories.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--split_seed", type=int, default=20260729)
    parser.add_argument(
        "--aesthetic_score",
        type=float,
        default=5.0,
        help="Constant metadata score; the input manifest is already filtered at >=4.5.",
    )
    parser.add_argument(
        "--alpha_threshold",
        type=float,
        default=0.8,
        help="Require condition images to contain foreground alpha above this normalized threshold.",
    )
    parser.add_argument(
        "--condition_mode",
        choices=["reference", "symlink", "copy"],
        default="reference",
        help="Reference absolute source images, or materialize them under renders_cond.",
    )
    parser.add_argument(
        "--validate",
        choices=["paths", "headers", "full"],
        default="headers",
        help="Validation depth for teacher NPZ files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel I/O workers used to validate and index samples.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "sample_id" not in rows[0]:
        raise ValueError("Manifest must contain at least one row and a sample_id column")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def teacher_paths(teacher_dir: Path, sample_id: str) -> tuple[Path, Path, Path]:
    return (
        teacher_dir / "ss_latents" / SS_LATENT_MODEL / f"{sample_id}.npz",
        teacher_dir / "latents" / SLAT_MODEL / f"{sample_id}.npz",
        teacher_dir / "records" / f"{sample_id}.json",
    )


def validate_npz(ss_path: Path, slat_path: Path, mode: str) -> int:
    if mode == "paths":
        return -1
    with np.load(ss_path) as ss_data:
        if "mean" not in ss_data or ss_data["mean"].shape != (8, 16, 16, 16):
            shape = ss_data["mean"].shape if "mean" in ss_data else None
            raise ValueError(f"Invalid sparse latent mean in {ss_path}: {shape}")
        if mode == "full" and not np.isfinite(ss_data["mean"]).all():
            raise ValueError(f"Non-finite sparse latent: {ss_path}")

    with np.load(slat_path) as slat_data:
        if not {"coords", "feats"}.issubset(slat_data.files):
            raise ValueError(f"Missing coords/feats in {slat_path}")
        coords, feats = slat_data["coords"], slat_data["feats"]
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Invalid coords shape in {slat_path}: {coords.shape}")
        if feats.ndim != 2 or feats.shape != (coords.shape[0], 8):
            raise ValueError(f"Invalid feats shape in {slat_path}: {feats.shape}")
        if mode == "full":
            if not np.isfinite(feats).all():
                raise ValueError(f"Non-finite SLat features: {slat_path}")
            if len(coords) and (coords.min() < 0 or coords.max() >= 64):
                raise ValueError(f"SLat coordinates outside [0, 64): {slat_path}")
        return int(coords.shape[0])


def check_condition_image(path: Path, alpha_threshold: float) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Condition image not found: {path}")
    with Image.open(path) as image:
        if image.mode not in ("RGBA", "LA") and not (image.mode == "P" and "transparency" in image.info):
            raise ValueError(
                f"Condition image must retain alpha for the stock TRELLIS training loader: {path} ({image.mode})"
            )
        alpha = np.asarray(image.convert("RGBA").getchannel("A"))
        if not np.any(alpha > alpha_threshold * 255):
            raise ValueError(f"Condition image has no alpha above threshold {alpha_threshold}: {path}")


def materialize_conditions(
    sample_id: str,
    source_images: list[Path],
    output_dir: Path,
    mode: str,
    alpha_threshold: float,
) -> list[str]:
    condition_dir = output_dir / "renders_cond" / sample_id
    condition_dir.mkdir(parents=True, exist_ok=True)
    file_paths: list[str] = []
    for index, source in enumerate(source_images):
        check_condition_image(source, alpha_threshold)
        if mode == "reference":
            file_paths.append(str(source.resolve()))
            continue

        suffix = source.suffix.lower() or ".png"
        destination = condition_dir / f"view_{index:02d}{suffix}"
        if mode == "symlink":
            if destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise FileExistsError(f"Conflicting condition symlink: {destination}")
            elif destination.exists():
                raise FileExistsError(f"Condition destination exists and is not a symlink: {destination}")
            else:
                destination.symlink_to(source.resolve())
        else:
            if not destination.exists():
                from shutil import copy2

                copy2(source, destination)
        file_paths.append(destination.name)

    atomic_write_json(
        condition_dir / "transforms.json",
        {
            "sample_id": sample_id,
            "frames": [{"file_path": value} for value in file_paths],
        },
    )
    return file_paths


def split_groups(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> None:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    groups = sorted({str(row["car_id"]) for row in rows})
    groups.sort(key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest())
    val_count = min(len(groups), int(math.ceil(len(groups) * val_ratio))) if val_ratio else 0
    validation_groups = set(groups[:val_count])
    for row in rows:
        row["split"] = "val" if str(row["car_id"]) in validation_groups else "train"


def ensure_directory_link(root: Path, name: str, target: Path) -> None:
    link = root / name
    expected = target.resolve()
    if link.is_symlink():
        if link.resolve() != expected:
            raise FileExistsError(f"Conflicting symlink: {link} -> {link.resolve()}, expected {expected}")
        return
    if link.exists():
        if link.resolve() != expected:
            raise FileExistsError(f"Path already exists and does not reference {expected}: {link}")
        return
    link.symlink_to(os.path.relpath(expected, start=root), target_is_directory=True)


def link_teacher_directories(teacher_dir: Path, output_dir: Path) -> None:
    for name in ("ss_latents", "latents"):
        source = teacher_dir / name
        if not source.is_dir():
            raise FileNotFoundError(source)
        if teacher_dir.resolve() != output_dir.resolve():
            ensure_directory_link(output_dir, name, source)


def process_manifest_row(
    manifest_row: dict[str, str],
    teacher_dir: Path,
    output_dir: Path,
    condition_mode: str,
    alpha_threshold: float,
    validate: str,
    aesthetic_score: float,
) -> tuple[dict[str, Any] | None, dict[str, str] | None, int | None]:
    sample_id = manifest_row["sample_id"].strip()
    try:
        if Path(sample_id).name != sample_id:
            raise ValueError("sample_id is not filesystem-safe")
        ss_path, slat_path, record_path = teacher_paths(teacher_dir, sample_id)
        if not (ss_path.is_file() and slat_path.is_file() and record_path.is_file()):
            raise FileNotFoundError("teacher export is incomplete")
        record = read_json(record_path)
        if record.get("status") != "ok":
            raise ValueError(f"teacher record status is {record.get('status')!r}")
        source_images = [Path(value).expanduser().resolve() for value in record.get("selected_images", [])]
        if len(source_images) != 4:
            raise ValueError(f"expected exactly four condition images, got {len(source_images)}")
        materialize_conditions(
            sample_id,
            source_images,
            output_dir,
            condition_mode,
            alpha_threshold,
        )

        checked_voxels = validate_npz(ss_path, slat_path, validate)
        num_voxels = int(record.get("num_voxels", checked_voxels))
        if checked_voxels >= 0 and num_voxels != checked_voxels:
            raise ValueError(f"record/NPZ voxel mismatch: {num_voxels} vs {checked_voxels}")
        if num_voxels <= 0:
            raise ValueError(f"invalid num_voxels: {num_voxels}")

        car_id = str(record.get("car_id") or sample_id.split("__", 1)[0])
        accepted = {
            "sha256": sample_id,
            "car_id": car_id,
            "variant_id": str(record.get("variant_id", "")),
            "aesthetic_score": float(aesthetic_score),
            "num_voxels": num_voxels,
            "cond_rendered": True,
            SS_COLUMN: True,
            SLAT_COLUMN: True,
        }
        return accepted, None, num_voxels
    except Exception as exc:
        return None, {"sample_id": sample_id, "reason": str(exc)}, None


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    teacher_dir = Path(args.teacher_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else teacher_dir
    if not 0.0 <= args.alpha_threshold <= 1.0:
        raise ValueError("alpha_threshold must be in [0, 1]")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "renders_cond").mkdir(parents=True, exist_ok=True)
    link_teacher_directories(teacher_dir, output_dir)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    voxel_counts: list[int] = []
    rows = read_manifest(manifest_path)
    process = lambda row: process_manifest_row(
        row,
        teacher_dir,
        output_dir,
        args.condition_mode,
        args.alpha_threshold,
        args.validate,
        args.aesthetic_score,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for accepted_row, rejected_row, num_voxels in executor.map(process, rows):
            if accepted_row is not None:
                accepted.append(accepted_row)
                voxel_counts.append(num_voxels)
            else:
                rejected.append(rejected_row)

    if not accepted:
        raise RuntimeError("No complete teacher samples were accepted")
    split_groups(accepted, args.val_ratio, args.split_seed)

    fields = [
        "sha256",
        "car_id",
        "variant_id",
        "aesthetic_score",
        "num_voxels",
        "cond_rendered",
        SS_COLUMN,
        SLAT_COLUMN,
        "split",
    ]
    write_csv(output_dir / "metadata.csv", accepted, fields)
    for split in ("train", "val"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_rows = [row for row in accepted if row["split"] == split]
        write_csv(split_dir / "metadata.csv", split_rows, fields)
        for name in ("ss_latents", "latents", "renders_cond"):
            ensure_directory_link(split_dir, name, output_dir / name)

    if rejected:
        write_csv(output_dir / "rejected.csv", rejected, ["sample_id", "reason"])
    split_counts = Counter(row["split"] for row in accepted)
    voxel_array = np.asarray(voxel_counts, dtype=np.int64)
    report = {
        "manifest": str(manifest_path),
        "teacher_dir": str(teacher_dir),
        "output_dir": str(output_dir),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "split_counts": dict(split_counts),
        "unique_car_ids": len({row["car_id"] for row in accepted}),
        "condition_mode": args.condition_mode,
        "alpha_threshold": args.alpha_threshold,
        "workers": args.workers,
        "voxel_count": {
            "min": int(voxel_array.min()),
            "median": int(np.median(voxel_array)),
            "p95": int(np.percentile(voxel_array, 95)),
            "max": int(voxel_array.max()),
            "over_32768": int((voxel_array > 32768).sum()),
        },
    }
    atomic_write_json(output_dir / "dataset_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
