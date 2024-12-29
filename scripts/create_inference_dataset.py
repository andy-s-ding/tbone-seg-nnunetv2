"""
create_inference_dataset.py
"""

import os
import glob
import fnmatch
import random
import pickle
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def rename_file(idx, path, file_type):
    # Rename file into nnUNet format
    if file_type == 'image':
        renamed_file = os.path.join(path, "jhu_" + '{0:0>3}'.format(idx) + "_0000.nii.gz")
    elif file_type == 'label':
        renamed_file = os.path.join(path, "jhu_" + '{0:0>3}'.format(idx) + ".nii.gz")
    return renamed_file

def return_image(file_id, dirpath):
    # Return registered image path given corresponding subject ID
    return os.path.join(dirpath, file_id + ".nii.gz")

def main(argv):
    data_path = argv[0]
    save_dir = argv[1]
    try: input_datasplit = argv[2]
    except: input_datasplit = 'datasplit.pkl'
    try: csv_name = argv[3]
    except: csv_name = 'inference_mapping.csv'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with open(input_datasplit, 'rb') as pickle_in:
        original_datasplit = pickle.load(pickle_in)
    train_test_images = set(x for xs in original_datasplit.values() for x in xs)
    num_train_test_images = len(train_test_images)

    all_images = set(target.split('.nii.gz')[0] for target in os.listdir(data_path) if fnmatch.fnmatch(target, '[LR]T_*.nii.gz'))
    inference_images = all_images.difference(train_test_images)

    column_names = ['Original File', 'Mapping']
    mapping_df = pd.DataFrame(columns = column_names)

    print("Copying Inference Files Over...")
    for i, file in tqdm(enumerate(inference_images, num_train_test_images)): # Start numbering after train/test images
        orig_img = return_image(file, data_path)
        new_img = rename_file(i, save_dir, 'image')
        mapping_df = pd.concat([mapping_df, pd.DataFrame.from_records([{column_names[0]: [orig_img], column_names[1]: [new_img]}])])

        command = f"cp {orig_img} {new_img}"
        # print(command)
        os.system(command)

    mapping_df.to_csv(os.path.join('.', csv_name))
        
if __name__ == '__main__':
    main(sys.argv[1:])
    # usage: python3 create_inference_dataset.py ../niftis/ ../niftis/test