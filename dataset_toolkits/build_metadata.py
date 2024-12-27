import os
import shutil
import sys
import time
import importlib
import argparse
import pandas as pd 
from typing import Union, Optional, List
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
import utils3d


def build_metadata(dataset_name: str, opt: Union[edict, argparse.ArgumentParser], cmd_args: Optional[List] = None):
    """ 
    Build meta information about the dataset already processed & to be processed 
    Simple Wrapper to support call from other scripts and from command line (should have cmd_args)
    
    Args:
        dataset_name (str): name of the dataset to be processed
        opt (Union[edict, argparse.ArgumentParser]): options(call) or parser(cli)
        cmd_args (Optional[List], optional): cli args, NEEDed when called from command line. Defaults to None.
    """
    dataset_utils = importlib.import_module(f'datasets.{dataset_name}')
    if isinstance(opt, argparse.ArgumentParser):
        dataset_utils.add_args(parser)
        opt = parser.parse_args(cmd_args)
        opt = edict(vars(opt))
    
    output_dir = opt.get("output_dir", None)
    field = opt.get('field', 'all')
    from_file = opt.get('from_file', False)
    assert output_dir is not None, "Output directory MUST be provided."
    
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'merged_records'), exist_ok=True)

    field = field.split(',')
    timestamp = str(int(time.time()))

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        print('Loading previous metadata...')
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        metadata = dataset_utils.get_metadata(**opt)
    metadata.set_index('sha256', inplace=True)
    
    # merge downloaded
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('downloaded_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        if 'local_path' in metadata.columns:
            metadata.update(df, overwrite=True)
        else:
            metadata = metadata.join(df, on='sha256', how='left')
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
            
    # detect models
    image_models = []
    if os.path.exists(os.path.join(opt.output_dir, 'features')):
        image_models = os.listdir(os.path.join(opt.output_dir, 'features'))
    latent_models = []
    if os.path.exists(os.path.join(opt.output_dir, 'latents')):
        latent_models = os.listdir(os.path.join(opt.output_dir, 'latents'))
    ss_latent_models = []
    if os.path.exists(os.path.join(opt.output_dir, 'ss_latents')):
        ss_latent_models = os.listdir(os.path.join(opt.output_dir, 'ss_latents'))
    print(f'Image models: {image_models}')
    print(f'Latent models: {latent_models}')
    print(f'Sparse Structure latent models: {ss_latent_models}')

    if 'rendered' not in metadata.columns:
        metadata['rendered'] = [False] * len(metadata)
    if 'voxelized' not in metadata.columns:
        metadata['voxelized'] = [False] * len(metadata)
    if 'num_voxels' not in metadata.columns:
        metadata['num_voxels'] = [0] * len(metadata)
    if 'cond_rendered' not in metadata.columns:
        metadata['cond_rendered'] = [False] * len(metadata)
    for model in image_models:
        if f'feature_{model}' not in metadata.columns:
            metadata[f'feature_{model}'] = [False] * len(metadata)
    for model in latent_models:
        if f'latent_{model}' not in metadata.columns:
            metadata[f'latent_{model}'] = [False] * len(metadata)
    for model in ss_latent_models:
        if f'ss_latent_{model}' not in metadata.columns:
            metadata[f'ss_latent_{model}'] = [False] * len(metadata)
    
    # merge rendered
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('rendered_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    # merge voxelized
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('voxelized_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    # merge cond_rendered
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('cond_rendered_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    # merge features
    for model in image_models:
        df_files = [f for f in os.listdir(opt.output_dir) if f.startswith(f'feature_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
                
    # merge latents
    for model in latent_models:
        df_files = [f for f in os.listdir(opt.output_dir) if f.startswith(f'latent_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
                
    # merge sparse structure latents
    for model in ss_latent_models:
        df_files = [f for f in os.listdir(opt.output_dir) if f.startswith(f'ss_latent_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    need_process = lambda key: key in field or field == ['all']

    # build metadata from files
    if from_file:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(metadata), desc="Building metadata") as pbar:
            def worker(sha256):
                try:
                    if need_process('rendered') and metadata.loc[sha256, 'rendered'] == False and \
                        os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
                        metadata.loc[sha256, 'rendered'] = True
                    if need_process('voxelized') and metadata.loc[sha256, 'rendered'] == True and metadata.loc[sha256, 'voxelized'] == False and \
                        os.path.exists(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')):
                        try:
                            pts = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
                            metadata.loc[sha256, 'voxelized'] = True
                            metadata.loc[sha256, 'num_voxels'] = len(pts)
                        except Exception as e:
                            pass
                    if need_process('cond_rendered') and metadata.loc[sha256, 'cond_rendered'] == False and \
                        os.path.exists(os.path.join(opt.output_dir, 'renders_cond', sha256, 'transforms.json')):
                        metadata.loc[sha256, 'cond_rendered'] = True
                    for model in image_models:
                        if need_process(f'feature_{model}') and \
                            metadata.loc[sha256, f'feature_{model}'] == False and \
                            metadata.loc[sha256, 'rendered'] == True and \
                            metadata.loc[sha256, 'voxelized'] == True and \
                            os.path.exists(os.path.join(opt.output_dir, 'features', model, f'{sha256}.npz')):
                            metadata.loc[sha256, f'feature_{model}'] = True
                    for model in latent_models:
                        if need_process(f'latent_{model}') and \
                            metadata.loc[sha256, f'latent_{model}'] == False and \
                            metadata.loc[sha256, 'rendered'] == True and \
                            metadata.loc[sha256, 'voxelized'] == True and \
                            os.path.exists(os.path.join(opt.output_dir, 'latents', model, f'{sha256}.npz')):
                            metadata.loc[sha256, f'latent_{model}'] = True
                    for model in ss_latent_models:
                        if need_process(f'ss_latent_{model}') and \
                            metadata.loc[sha256, f'ss_latent_{model}'] == False and \
                            metadata.loc[sha256, 'voxelized'] == True and \
                            os.path.exists(os.path.join(opt.output_dir, 'ss_latents', model, f'{sha256}.npz')):
                            metadata.loc[sha256, f'ss_latent_{model}'] = True
                    pbar.update()
                except Exception as e:
                    print(f'Error processing {sha256}: {e}')
                    pbar.update()
            
            executor.map(worker, metadata.index)
            executor.shutdown(wait=True)

    # statistics
    metadata.to_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    num_downloaded = metadata['local_path'].count() if 'local_path' in metadata.columns else 0
    with open(os.path.join(opt.output_dir, 'statistics.txt'), 'w') as f:
        f.write('Statistics:\n')
        f.write(f'  - Number of assets: {len(metadata)}\n')
        f.write(f'  - Number of assets downloaded: {num_downloaded}\n')
        f.write(f'  - Number of assets rendered: {metadata["rendered"].sum()}\n')
        f.write(f'  - Number of assets voxelized: {metadata["voxelized"].sum()}\n')
        if len(image_models) != 0:
            f.write(f'  - Number of assets with image features extracted:\n')
            for model in image_models:
                f.write(f'    - {model}: {metadata[f"feature_{model}"].sum()}\n')
        if len(latent_models) != 0:
            f.write(f'  - Number of assets with latents extracted:\n')
            for model in latent_models:
                f.write(f'    - {model}: {metadata[f"latent_{model}"].sum()}\n')
        if len(ss_latent_models) != 0:
            f.write(f'  - Number of assets with sparse structure latents extracted:\n')
            for model in ss_latent_models:
                f.write(f'    - {model}: {metadata[f"ss_latent_{model}"].sum()}\n')
        f.write(f'  - Number of assets with captions: {metadata["captions"].count()}\n')
        f.write(f'  - Number of assets with image conditions: {metadata["cond_rendered"].sum()}\n')
        
    with open(os.path.join(opt.output_dir, 'statistics.txt'), 'r') as f:
        print(f.read())


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--field', type=str, default='all',
                        help='Fields to process, separated by commas')
    parser.add_argument('--from_file', action='store_true',
                        help='Build metadata from file instead of from records of processings.' +
                             'Useful when some processing fail to generate records but file already exists.')

    build_metadata(dataset_name, parser, cmd_args=sys.argv[2:])
