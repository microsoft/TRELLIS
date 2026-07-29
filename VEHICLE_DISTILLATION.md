# Vehicle Four-to-One View Distillation

This workflow distills the selected four-view TRELLIS results into the two
image-conditioned Flow Transformers while presenting one randomly selected
view as the training condition.

## Source data

The filtered manifest currently contains 12,896 samples from 3,658 `car_id`
groups:

```text
/cluster/home/liaolw/dataset/3drealcar_zzj/
  car_orbit_quarters_shape_filter/
    car_orbit_quarters_top_aspect_ge_2p0_aesthetic_ge4p5_links/manifest.csv
```

Each manifest target contains a `meta.json` written by
`generate_3drealcar_gaussian.py`. It records the exact four selected images,
seed, sampling steps, CFG strengths, preprocessing mode, and multi-image mode.
The exporter replays these settings and stops before Gaussian decoding.

The exported targets are:

```text
ss_latents/ss_enc_conv3d_16l8_fp16/<sample_id>.npz
  mean: [8, 16, 16, 16]

latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/<sample_id>.npz
  coords: [N, 3]
  feats:  [N, 8]
```

`feats` stores the denormalized SLat returned by `sample_slat()`. The existing
TRELLIS dataset applies the official normalization during loading.

## Export on node10

```bash
ssh node10
source ~/.bashrc
conda activate trellis
cd /cluster/home/liaolw/code/zzj_code/TRELLIS

bash dataset_toolkits/run_vehicle_teacher_export.sh
```

The launcher assigns one exporter to each available A40 GPU and writes to:

```text
/cluster/home/liaolw/dataset/3drealcar_trellis
```

The command is resumable. Completed samples are skipped. Per-rank logs and
failures are stored under `logs/` and `failures_rank*.jsonl`. Set
`WORLD_SIZE` when fewer than four GPUs are available; for example, the initial
export used GPUs 0-2 because GPU 3 was occupied:

```bash
WORLD_SIZE=3 bash dataset_toolkits/run_vehicle_teacher_export.sh
```

For a small check before the full export:

```bash
CUDA_VISIBLE_DEVICES=0 python -u dataset_toolkits/export_vehicle_teacher_latents.py \
  --manifest /cluster/home/liaolw/dataset/3drealcar_zzj/car_orbit_quarters_shape_filter/car_orbit_quarters_top_aspect_ge_2p0_aesthetic_ge4p5_links/manifest.csv \
  --output_dir /tmp/vehicle_distill_check \
  --max_samples 4

python dataset_toolkits/build_vehicle_distillation_dataset.py \
  --manifest /cluster/home/liaolw/dataset/3drealcar_zzj/car_orbit_quarters_shape_filter/car_orbit_quarters_top_aspect_ge_2p0_aesthetic_ge4p5_links/manifest.csv \
  --teacher_dir /tmp/vehicle_distill_check \
  --validate full \
  --val_ratio 0
```

The finalizer creates `train/` and `val/` views of the dataset. Splitting is
performed by `car_id`, so variants of the same source group cannot leak across
the boundary. Condition images remain referenced by absolute path by default;
use `--condition_mode symlink` or `copy` to materialize them.

## Fine-tuning

The configs initialize directly from the official local safetensors weights.
They target two A40 48GB GPUs with batch size 1 per GPU, global batch size 2,
learning rate `1e-5`, and 20,000 optimization steps.
The vehicle configs use an alpha threshold of `0.8`, matching
`TrellisImageTo3DPipeline.preprocess_image()` at inference time.

Start with SLat Flow because it controls most vehicle appearance and local
detail:

```bash
python train.py \
  --config configs/generation/vehicle_distill_slat_flow_img_dit_L_64l8p2_fp16.json \
  --output_dir outputs/vehicle_distill_slat \
  --data_dir /cluster/home/liaolw/dataset/3drealcar_trellis/train \
  --num_gpus 2 \
  --auto_retry 0
```

Then fine-tune Sparse Structure Flow if single-view silhouette and global car
geometry still trail the four-view teacher:

```bash
python train.py \
  --config configs/generation/vehicle_distill_ss_flow_img_dit_L_16l8_fp16.json \
  --output_dir outputs/vehicle_distill_ss \
  --data_dir /cluster/home/liaolw/dataset/3drealcar_trellis/train \
  --num_gpus 2 \
  --auto_retry 0
```

Use the EMA checkpoints for evaluation. A single image cannot determine truly
occluded details; the student learns the vehicle-domain prior represented by
the 12,896 four-view teacher targets rather than reproducing hidden evidence
that is absent from its input.

## Two-GPU runtime

Short DDP benchmarks on two node10 A40 GPUs, using the final train split and
the production data-loader settings, measured 0.744 seconds per SLat Flow
optimizer step and 0.704 seconds per Sparse Structure Flow step. At 20,000
steps per model, pure optimization takes about 4.13 and 3.91 hours. Allow
9-10 hours to train both models sequentially after including initialization,
checkpoint writes, and snapshots. Each model sees 40,000 samples, equivalent
to about 3.27 passes over the 12,248-sample train split.
