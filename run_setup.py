# Libraries
import os
import numpy as np
import math
import shutil
from tqdm import tqdm

# Constants
PATH_PREFIX = 'F:\SANYU\Projects\leukemia-classification\dataset'
SPLIT_PATH_PREFIX = 'F:\SANYU\Projects\leukemia-classification\data-split'
CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05
TEST_SPLIT = 0.05

# Util functions
def get_class_path(index):
    return os.path.join(PATH_PREFIX, CLASS_NAMES[index])

def get_class_split_path(index):
    return os.path.join(SPLIT_PATH_PREFIX, CLASS_NAMES[index])
    
for i in range(len(CLASS_NAMES)):
    print(f"Splitting class - {CLASS_NAMES[i]}")
    
    # Fetching all files inside this class.
    filenames = os.listdir(get_class_path(i))
    
    # Shuffling file names and splitting
    np.random.shuffle(filenames)
    train_len = math.floor(TRAIN_SPLIT * len(filenames))
    val_len = math.floor(VAL_SPLIT * len(filenames))
    train_names, val_names, test_names = (
        filenames[: train_len],
        filenames[train_len: train_len + val_len],
        filenames[train_len + val_len:]
    )
    
    # Creating destination path folders if required.
    dest_path = get_class_split_path(i)
    os.makedirs(dest_path, exist_ok=True) 
    
    # Copying files to dedicated folder after splitting
    for split in (train_names, val_names, test_names):
        print("Copying split")
        for name in tqdm(split):
            src_file_path = os.path.join(get_class_path(i), name)
            dest_file_path = os.path.join(get_class_split_path(i), name)
            shutil.copyfile(src_file_path, dest_file_path)