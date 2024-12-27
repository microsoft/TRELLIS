# TRELLIS-500K

TRELLIS-500K is a dataset of 500K 3D assets curated from [Objaverse(XL)](https://objaverse.allenai.org/), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [HSSD](https://huggingface.co/datasets/hssd/hssd-models), and [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k), filtered based on aesthetic scores.
This dataset serves for 3D generation tasks.

The dataset is provided as csv files containing the 3D assets' metadata.

## Dataset Statistics

The following table summarizes the dataset's filtering and composition:

***NOTE: Some of the 3D assets lack text captions. Please filter out such assets if captions are required.***
| Source | Aesthetic Score Threshold | Filtered Size | With Captions |
|:-:|:-:|:-:|:-:|
| ObjaverseXL (sketchfab) | 5.5 | 168307 | 167638 |
| ObjaverseXL (github) | 5.5 | 311843 | 306790 |
| ABO | 4.5 | 4485 | 4390 |
| 3D-FUTURE | 4.5 | 9472 | 9291 |
| HSSD | 4.5 | 6670 | 6661 |
| All (training set) | - | 500777 | 494770 |
| Toys4k (evaluation set) | 4.5 | 3229 | 3180 |

## Dataset Location

The dataset is hosted on Hugging Face Datasets. You can preview the dataset at

[https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K)

There is no need to download the csv files manually. We provide toolkits to load and prepare the dataset.

## Dataset Toolkits

We provide [toolkits](dataset_toolkits) for data preparation.

### Step 1: Install Dependencies

```
. ./dataset_toolkits/setup.sh
```

### Step 2: Load Metadata

First, we need to load the metadata of the dataset.

```
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <OUTPUT_DIR> [--source <SOURCE>]
```

- `SUBSET`: The subset of the dataset to load. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `SOURCE`: Required if `SUBSET` is `ObjaverseXL`. Options are `sketchfab` and `github`.

For example, to load the metadata of the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 3: Download Data

Next, we need to download the 3D assets.

```
python dataset_toolkits/download.py <SUBSET> --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to download. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.

You can also specify the `RANK` and `WORLD_SIZE` of the current process if you are using multiple nodes for data preparation.

For example, to download the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run: 

***NOTE: The example command below sets a large `WORLD_SIZE` for demonstration purposes. Only a small portion of the dataset will be downloaded.***

```
python dataset_toolkits/download.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --world_size 160000
```

Some datasets may require interactive login to Hugging Face or manual downloading. Please follow the instructions given by the toolkits.



### Step 4: Render Multiview Images

Multiview images can be rendered with:

```
python dataset_toolkits/render.py <SUBSET> --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to render. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `NUM_VIEWS`: The number of views to render. Default is 150.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to render the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/render.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```



### Step 5: Voxelize 3D Models

We can voxelize the 3D models with:

```
python dataset_toolkits/voxelize.py <SUBSET> --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to voxelize. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to voxelize the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:
```
python dataset_toolkits/voxelize.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 6: Extract DINO Features

To prepare the training data for SLat VAE, we need to extract DINO features from multiview images and aggregate them into sparse voxel grids.

```
python dataset_toolkits/extract_features.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.


For example, to extract DINO features from the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/extract_feature.py --output_dir datasets/ObjaverseXL_sketchfab
```

Now the metadata file should be updated manually:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 7: Encode Sparse Structures

Encoding the sparse structures into latents to train the first stage generator:

```
python dataset_toolkits/encode_ss_latent.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to encode the sparse structures into latents for the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/encode_ss_latent.py --output_dir datasets/ObjaverseXL_sketchfab
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 8: Encode SLat

Encoding SLat for second stage generator training:

```
python dataset_toolkits/encode_latent.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to encode SLat for the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/encode_latent.py --output_dir datasets/ObjaverseXL_sketchfab
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 9: Render Image Conditions

To train the image conditioned generator, we need to render image conditions with augmented views.

```
python dataset_toolkits/render_cond.py <SUBSET> --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to render. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `NUM_VIEWS`: The number of views to render. Default is 24.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to render image conditions for the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### All in ONE Script 
Merge all above procedures into a single shell script:
``` 
bash dataset_toolkits/dataset_pipe.sh
```
