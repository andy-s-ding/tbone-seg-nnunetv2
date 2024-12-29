"""
convert_to_nnunet_dataset.py
"""

import os
import glob
import fnmatch
import sys
import pandas as pd
from tqdm import tqdm
import argparse
import shutil

def rename_file(idx, path, file_type, prefix="jhu"):
    # Rename file into nnUNet format
    if file_type == 'image':
        renamed_file = os.path.join(path, f"{prefix}_" + '{0:0>3}'.format(idx) + "_0000.nii.gz")
    elif file_type == 'label':
        renamed_file = os.path.join(path, f"{prefix}_" + '{0:0>3}'.format(idx) + ".nii.gz")
    return renamed_file

def return_image(file_id, dirpath):
    # Return registered image path given corresponding subject ID
    return os.path.join(dirpath, file_id + ".nii.gz")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a folder of NIFTI images to a dataset compatible with nnUNet.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input directory containing images.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output directory containing images.')
    parser.add_argument('-csv', '--csv_name', type=str, required=False, help='CSV filename to map input images to output images', default="inference_mapping")
    parser.add_argument('-p', '--prefix', type=str, required=False, help='Prefix for images in output dataset', default="jhu")
    parser.add_argument('-idx', '--index', type=int, required=False, help='Starting index number for images', default=0)
    args = vars(parser.parse_args())

    input_dir = args['input']
    output_dir = args['output']
    csv_name = f"{args['csv_name']}.csv"
    start_index = args['index']

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    inference_images = [target.split('.nii.gz')[0] for target in os.listdir(input_dir) if fnmatch.fnmatch(target, '*.nii.gz')]

    column_names = ['Original File', 'Mapping']
    mapping_df = pd.DataFrame(columns = column_names)

    print("Copying Inference Files Over...")
    for i, file in tqdm(enumerate(inference_images)):
        orig_img = return_image(file, input_dir)
        new_img = rename_file(start_index + i, output_dir, 'image')
        mapping_df = pd.concat([mapping_df, pd.DataFrame.from_records([{column_names[0]: [orig_img], column_names[1]: [new_img]}])])
        try:
            shutil.copy(orig_img, new_img)
        except Exception as e:
            print(f"Error copying {orig_img} to {new_img}: {e}")

    mapping_df.to_csv(os.path.join(output_dir, csv_name))

    # usage: python3 convert_to_nnunet_dataset.py -i /media/andyding/172A-29C2/Ear_IRDynaCT/ -o /media/andyding/172A-29C2/Ear_IRDynaCT_nnUNet/