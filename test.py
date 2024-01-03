import os
import lmdb
import cv2 as cv
import numpy as np

from tqdm import tqdm
from src.utils import LMDB

def get_total_buffer_size(env):
    total_size = 0
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            total_size += len(value)
    # env.close()
    return total_size 

def main(source_file, save_dir):
    _, filename = os.path.split(source_file)
    print(source_file)
    env = lmdb.open(source_file)
    
    
    old_meta = []
    with open(source_file + "/meta_info_0.txt", 'r') as file:
        for line in file.readlines():
            old_meta.append(line.strip())
    
    total_size = get_total_buffer_size(env)
    old_txn = env.begin()
    
    lmdb_path = os.path.join(save_dir,  filename)
  
    lmdb_env = lmdb.open(lmdb_path, map_size=total_size * 4)
    meta_info = []
    with lmdb_env.begin(write=True) as txn:
        for line in tqdm(old_meta, desc='lmdb'):
            line = line.split(' ')[0].split('.')[0]
            key = line.encode('ascii')
            buffer = old_txn.get(key)
            img_np = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv.IMREAD_COLOR)
            img_rgb = img_np[:, :, ::-1]
            txn.put(key, img_rgb.tobytes())
            meta_info.append(line + '-' + f"{img_rgb.shape}")
    
    env.close()
    lmdb_env.close()
    
    with open(os.path.join(lmdb_path, 'meta_info.txt'),'w') as meta:
        for info in meta_info:
            meta.write(info + '\n')
    
    print(f"=>=> the filepath of ldmb is {lmdb_path}")

if __name__ == '__main__':
    save_dir = "/data/dataset/low_level/sidd"  
    files = [
        '/data/dataset/low_level/sidd_old/val_gt.lmdb',
        '/data/dataset/low_level/sidd_old/val_noisy.lmdb',
        '/data/dataset/low_level/sidd_old/train_gt.lmdb',
        '/data/dataset/low_level/sidd_old/train_noisy.lmdb',
    ]
    
    for file in files:
        main(source_file=file,
             save_dir=save_dir)
    