#!/usr/bin/env python3
"""Export four-view TRELLIS teacher latents for single-view distillation.

The source manifest is expected to map a filesystem-safe ``sample_id`` to a
TRELLIS generation directory containing ``meta.json``. The metadata produced
by ``generate_3drealcar_gaussian.py`` records the exact four input images and
sampling parameters, so this script can reproduce the teacher trajectory
without decoding another Gaussian PLY.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SS_LATENT_MODEL = "ss_enc_conv3d_16l8_fp16"
SLAT_MODEL = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce selected four-view teachers and save both Flow training targets."
    )
    parser.add_argument("--manifest", required=True, help="Filtered manifest.csv containing sample_id and target.")
    parser.add_argument("--output_dir", required=True, help="Destination TRELLIS training-data root.")
    parser.add_argument(
        "--model_name",
        default=None,
        help="TRELLIS-image-large path. Defaults to model_name in the first valid source meta.json.",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0, help="Global debug limit; 0 means all rows.")
    parser.add_argument("--overwrite", action="store_true")
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


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"sample_id", "target"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return rows


def source_meta_path(row: dict[str, str]) -> Path:
    return Path(row["target"]).expanduser() / "meta.json"


def validate_sample_id(sample_id: str) -> None:
    if not sample_id or Path(sample_id).name != sample_id or sample_id in {".", ".."}:
        raise ValueError(f"sample_id must be one filesystem-safe component: {sample_id!r}")


def resolve_selected_images(meta: dict[str, Any]) -> list[Path]:
    image_root = Path(meta["input_image_dir"]).expanduser()
    paths = []
    for value in meta.get("selected_images", []):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = image_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Condition image not found: {path}")
        paths.append(path)
    if len(paths) < 2:
        raise ValueError(f"Expected a multi-view teacher, got {len(paths)} selected image(s)")
    return paths


def background_rgb(name: str) -> tuple[int, int, int]:
    return {
        "white": (255, 255, 255),
        "gray": (230, 230, 230),
        "black": (0, 0, 0),
    }[name]


def prepare_custom_image(path: Path, meta: dict[str, Any]) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    array = np.asarray(image).astype(np.float32)
    alpha = array[..., 3]
    alpha_f = alpha[..., None] / 255.0
    bg = np.asarray(background_rgb(meta.get("bg_color", "white")), dtype=np.float32)
    rgb = array[..., :3] * alpha_f + bg.reshape(1, 1, 3) * (1.0 - alpha_f)
    rgb_image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    threshold = int(meta.get("alpha_threshold", 10))
    ys, xs = np.where(alpha > threshold)
    if len(xs):
        cx = (float(xs.min()) + float(xs.max())) * 0.5
        cy = (float(ys.min()) + float(ys.max())) * 0.5
        size = max(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
        size *= float(meta.get("crop_scale", 1.2))
        left, top = int(round(cx - size * 0.5)), int(round(cy - size * 0.5))
        right, bottom = int(round(cx + size * 0.5)), int(round(cy + size * 0.5))
        side = max(right - left, bottom - top, 1)
        canvas = Image.new("RGB", (side, side), background_rgb(meta.get("bg_color", "white")))
        src_box = (max(0, left), max(0, top), min(rgb_image.width, right), min(rgb_image.height, bottom))
        if src_box[2] > src_box[0] and src_box[3] > src_box[1]:
            canvas.paste(rgb_image.crop(src_box), (src_box[0] - left, src_box[1] - top))
        rgb_image = canvas
    else:
        side = max(rgb_image.size)
        canvas = Image.new("RGB", (side, side), background_rgb(meta.get("bg_color", "white")))
        canvas.paste(rgb_image, ((side - rgb_image.width) // 2, (side - rgb_image.height) // 2))
        rgb_image = canvas

    size = int(meta.get("prepared_size", 518))
    return rgb_image.resize((size, size), Image.Resampling.LANCZOS)


def load_teacher_images(paths: list[Path], meta: dict[str, Any]) -> tuple[list[Image.Image], bool]:
    preprocess_mode = meta.get("preprocess_mode", "trellis")
    if preprocess_mode == "custom":
        return [prepare_custom_image(path, meta) for path in paths], False
    if preprocess_mode != "trellis":
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    images = []
    for path in paths:
        with Image.open(path) as image:
            has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
            images.append(image.convert("RGBA" if has_alpha else "RGB").copy())
    return images, True


def output_paths(output_dir: Path, sample_id: str) -> tuple[Path, Path, Path]:
    ss_path = output_dir / "ss_latents" / SS_LATENT_MODEL / f"{sample_id}.npz"
    slat_path = output_dir / "latents" / SLAT_MODEL / f"{sample_id}.npz"
    record_path = output_dir / "records" / f"{sample_id}.json"
    return ss_path, slat_path, record_path


def output_complete(output_dir: Path, sample_id: str) -> bool:
    ss_path, slat_path, record_path = output_paths(output_dir, sample_id)
    if not (ss_path.is_file() and slat_path.is_file() and record_path.is_file()):
        return False
    try:
        return read_json(record_path).get("status") == "ok"
    except Exception:
        return False


def infer_model_name(rows: list[dict[str, str]]) -> str:
    for row in rows:
        path = source_meta_path(row)
        if not path.is_file():
            continue
        model_name = read_json(path).get("model_name")
        if model_name:
            return str(model_name)
    raise ValueError("Could not infer model_name from source meta.json files; pass --model_name explicitly")


def append_failure(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("Require world_size >= 1 and 0 <= rank < world_size")

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    model_name = args.model_name or infer_model_name(rows)
    shard = rows[args.rank :: args.world_size]

    print(f"[rank {args.rank}] manifest rows: {len(rows)}; shard rows: {len(shard)}")
    print(f"[rank {args.rank}] model: {model_name}")
    print(f"[rank {args.rank}] output: {output_dir}")

    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
    for decoder_name in ("slat_decoder_gs", "slat_decoder_rf", "slat_decoder_mesh"):
        pipeline.models.pop(decoder_name, None)
    pipeline.cuda()

    summary = {"ok": 0, "skipped": 0, "failed": 0, "rank": args.rank, "world_size": args.world_size}
    failure_path = output_dir / f"failures_rank{args.rank:03d}.jsonl"
    start = time.time()

    for index, row in enumerate(shard, start=1):
        sample_id = row["sample_id"].strip()
        try:
            validate_sample_id(sample_id)
            if not args.overwrite and output_complete(output_dir, sample_id):
                summary["skipped"] += 1
                continue

            meta_path = source_meta_path(row)
            meta = read_json(meta_path)
            if meta.get("status") != "ok":
                raise ValueError(f"Source generation status is {meta.get('status')!r}")
            selected_paths = resolve_selected_images(meta)
            images, preprocess_image = load_teacher_images(selected_paths, meta)
            trellis_args = meta.get("trellis", {})

            outputs = pipeline.run_multi_image(
                images,
                seed=int(meta.get("seed", 1)),
                sparse_structure_sampler_params={
                    "steps": int(trellis_args.get("sparse_steps", 12)),
                    "cfg_strength": float(trellis_args.get("sparse_cfg", 7.5)),
                },
                slat_sampler_params={
                    "steps": int(trellis_args.get("slat_steps", 12)),
                    "cfg_strength": float(trellis_args.get("slat_cfg", 3.0)),
                },
                formats=[],
                preprocess_image=bool(trellis_args.get("preprocess_image", preprocess_image)),
                mode=meta.get("multi_image_mode", "stochastic"),
                return_intermediates=True,
            )
            intermediates = outputs["_intermediates"]
            z_s = intermediates["sparse_structure_latent"]
            coords = intermediates["coords"]
            slat = intermediates["slat"]

            if z_s.ndim != 5 or z_s.shape[0] != 1:
                raise ValueError(f"Unexpected sparse latent shape: {tuple(z_s.shape)}")
            if coords.ndim != 2 or coords.shape[1] != 4:
                raise ValueError(f"Unexpected coords shape: {tuple(coords.shape)}")
            if slat.feats.ndim != 2 or slat.coords.shape != coords.shape:
                raise ValueError(
                    f"Unexpected SLat shapes: feats={tuple(slat.feats.shape)}, coords={tuple(slat.coords.shape)}"
                )

            ss_path, slat_path, record_path = output_paths(output_dir, sample_id)
            ss_array = z_s[0].detach().float().cpu().numpy()
            coord_array = slat.coords[:, 1:].detach().cpu().numpy().astype(np.uint8)
            feat_array = slat.feats.detach().float().cpu().numpy()
            if not (np.isfinite(ss_array).all() and np.isfinite(feat_array).all()):
                raise ValueError("Teacher latents contain non-finite values")

            atomic_save_npz(ss_path, mean=ss_array)
            atomic_save_npz(slat_path, coords=coord_array, feats=feat_array)
            atomic_write_json(
                record_path,
                {
                    "status": "ok",
                    "sample_id": sample_id,
                    "car_id": str(meta.get("car_id", sample_id.split("__", 1)[0])),
                    "variant_id": str(meta.get("variant_id", "")),
                    "source_meta": str(meta_path),
                    "selected_images": [str(path) for path in selected_paths],
                    "model_name": model_name,
                    "seed": int(meta.get("seed", 1)),
                    "multi_image_mode": meta.get("multi_image_mode", "stochastic"),
                    "trellis": trellis_args,
                    "sparse_structure_latent_shape": list(ss_array.shape),
                    "coords_shape": list(coord_array.shape),
                    "slat_feats_shape": list(feat_array.shape),
                    "num_voxels": int(coord_array.shape[0]),
                },
            )
            summary["ok"] += 1
            if index == 1 or index % 50 == 0:
                elapsed = max(time.time() - start, 1e-6)
                print(
                    f"[rank {args.rank}] {index}/{len(shard)} {sample_id} "
                    f"voxels={len(coord_array)} rate={index / elapsed:.3f} samples/s",
                    flush=True,
                )
        except Exception as exc:
            summary["failed"] += 1
            append_failure(
                failure_path,
                {
                    "sample_id": sample_id,
                    "target": row.get("target"),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[rank {args.rank}] failed {sample_id}: {exc}", file=sys.stderr, flush=True)
            torch.cuda.empty_cache()

    summary["elapsed_sec"] = round(time.time() - start, 3)
    atomic_write_json(output_dir / f"summary_rank{args.rank:03d}.json", summary)
    print(f"[rank {args.rank}] summary: {summary}")
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
