set -e 
set -x 
# install requirements and depdendencies 
. ./dataset_toolkits/setup.sh

export DATASET_NAME=ObjaverseXL
export DATASET_SOURCE=sketchfab
export OUTPUT_DIR=datasets/ObjaverseXL_sketchfab
export RANK=0
export WORLD_SIZE=1600
export MAX_WORKERS=6

# initially build meta data of the specified dataset
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --source ${DATASET_SOURCE} --output_dir ${OUTPUT_DIR}

# download dataset
python dataset_toolkits/download.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --world_size ${WORLD_SIZE}

# render multi-view images with blender (e.g. 150 views)
python dataset_toolkits/render.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --max_workers ${MAX_WORKERS}

# voxelization (based on open3d)
python dataset_toolkits/voxelize.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR}

# extract DINOv2 features
python dataset_toolkits/extract_feature.py --output_dir ${OUTPUT_DIR}
# update metadata
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --source ${DATASET_SOURCE} --output_dir ${OUTPUT_DIR}

# extract SparseStructure Latents
python dataset_toolkits/encode_ss_latent.py --output_dir ${OUTPUT_DIR}
# update metadata
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --source ${DATASET_SOURCE} --output_dir ${OUTPUT_DIR}

# encode to SLATS
python dataset_toolkits/encode_latent.py --output_dir ${OUTPUT_DIR}
# update metadata manually 
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --source ${DATASET_SOURCE} --output_dir ${OUTPUT_DIR} 

# render condition images
python dataset_toolkits/render_cond.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR}
