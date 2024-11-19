import os
import glob
import random
import pickle
import sys
import pickle as pkl
import argparse
import datetime
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple
from batchgenerators.utilities.file_and_folder_operations import *
from nifti_resampler import resample_volume_from_reference


# Establish labels
label_key = {
    "background": 0,
    "Bone": 1,
    "Malleus": 2,
    "Incus": 3,
    "Stapes": 4,
    "Bony_Labyrinth": 5,
    "IAC": 6,
    "Superior_Vestibular_Nerve": 7,
    "Inferior_Vestibular_Nerve": 8,
    "Cochlear_Nerve": 9,
    "Facial_Nerve": 10,
    "Chorda_Tympani": 11,
    "ICA": 12,
    "Sigmoid_Sinus": 13,
    "Dura": 14,
    "Vestibular_Aqueduct": 15,
    "Mandible": 16,
    "EAC": 17
}

def rename_file(idx, path, file_type):
    # Rename file into nnUNet format
    if file_type == 'image':
        renamed_file = os.path.join(path, "jhu_" + '{0:0>3}'.format(idx) + "_0000.nii.gz")
    elif file_type == 'label':
        renamed_file = os.path.join(path, "jhu_" + '{0:0>3}'.format(idx) + ".nii.gz")
    return renamed_file

def return_label(file_id, dirpath=None):
    # Return label path given corresponding image file
    return os.path.join(dirpath, "Segmentation_" + file_id + ".nii.gz")

def return_image(file_id, dirpath):
    # Return registered image path given corresponding subject ID
    return os.path.join(dirpath, file_id + ".nii.gz")

def get_identifiers_from_split_files(folder: str):
    # nnUNet generate_dataset_json helper function
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    breakpoint()
    return uniques

def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str = '.nii.gz',
                          regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None, reference: str = None, release: str = '0.0', license: str = "hands off!",
                          description: str = None,
                          overwrite_image_reader_writer: str = None, **kwargs):
    """
    Generates a dataset.json file in the output folder

    channel_names:
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }

        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

    num_training_cases: is used to double check all cases are there!

    file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
    segmentations!

    dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
    completeness and as a reminder that these would be great!

    overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
    BaseReaderWriter, place it into nnunet.imageio and reference it here by name

    kwargs: whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
                                                f"You need that."
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        'channel_names': channel_names,  # previously this was called 'modality'. I didn't like this so this is
        # channel_names now. Live with it.
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if license is not None:
        dataset_json['licence'] = license
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
    

class FilestructureArgParser(object):
    """Arg Parser for Filestructure File."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='ArgParser for Filestructure Setup')
        self.parser.add_argument('-i', '--input_dir', type=str, help='Directory with registered original files')
        self.parser.add_argument('-p', '--pickle', type=str, help='Path to pickle file',
                                 default='datasplit.pkl')
        self.parser.add_argument('-o', '--output_dir', type=str, help='Output directory for nnUNet')
        self.parser.add_argument('-d', '--dataset_num', type=str, help='Dataset #',
                                 default='101')

    def parse_args(self):
        args = self.parser.parse_args()

        # Save args to a JSON file
        json_save_dir = './datasplit_jsons'
        if not os.path.isdir(json_save_dir):
            os.mkdir(json_save_dir)
        date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(json_save_dir, 'tbone_{}'.format(date_string))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')
        args.save_dir = save_dir

        return args


def main(args):
    input_dir = args.input_dir
    csv_name = 'original_mapping.csv'
    pkl_path = args.pickle
    output_dir = args.output_dir
    
    # Load in datasplit.pkl
    with open(pkl_path, "rb") as pickle_in:
        split = pkl.load(pickle_in)
    train_files = split['Train']
    test_files = split['Test']
    print(f"There are {len(train_files)} and {len(test_files)} test files.")
    
    
    # Establish filenames
    dataset_dir = os.path.join(output_dir, "nnUNet_raw", f"Dataset{args.dataset_num}_TemporalBone")
    train_dir = os.path.join(dataset_dir, "imagesTr")
    train_label_dir = os.path.join(dataset_dir, "labelsTr")
    test_dir = os.path.join(dataset_dir, "imagesTs")
    test_label_dir = os.path.join(dataset_dir, "labelsTs")
    
    column_names = ['Original File', 'Mapping']
    mapping_df = pd.DataFrame(columns = column_names)
    
    # Check if files already exist, if not, then cp files into correct folders
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "nnUNet_raw"))
        os.mkdir(dataset_dir)
        os.mkdir(train_dir)
        os.mkdir(train_label_dir)
        os.mkdir(test_dir)
        os.mkdir(test_label_dir)

        print("Copying Train Files Over...")
        for i, file in tqdm(enumerate(train_files)):
            orig_img = return_image(file, input_dir)
            new_img = rename_file(i, train_dir, 'image')
            mapping_df = pd.concat([mapping_df, pd.DataFrame.from_records([{column_names[0]: [orig_img], column_names[1]: [new_img]}])])

            command = f"cp {orig_img} {new_img}" # Copy image volume
            # print(command)
            os.system(command)

            orig_label = return_label(file, dirpath=input_dir)
            new_label = rename_file(i, train_label_dir, 'label')
            resample_volume_from_reference(orig_label, orig_img, new_label, labelmap=True, verbose=True) # Resample label volume
            # command = f"cp {orig_label} {new_label}" # Copy label volume (preprocessing may throw error for disrepancies volume:label spacing)
            # print(command)
            # os.system(command)
        print("Copying Test Files Over...")
        for j, file in tqdm(enumerate(test_files, i+1)):
            orig_img = return_image(file, input_dir)
            new_img = rename_file(j, test_dir, 'image')
            mapping_df = pd.concat([mapping_df, pd.DataFrame.from_records([{column_names[0]: [orig_img], column_names[1]: [new_img]}])])

            command = f"cp {orig_img} {new_img}"
            # print(command)
            os.system(command)

            orig_label = return_label(file, dirpath=input_dir)
            new_label = rename_file(j, test_label_dir, 'label')
            resample_volume_from_reference(orig_label, orig_img, new_label, labelmap=True, verbose=True) # Resample label volume
            # command = f"cp {orig_label} {new_label}" # Copy label volume (preprocessing may throw error for disrepancies volume:label spacing)
            # print(command)
            # os.system(command)
                
        # Make dataset.json file
        mapping_df.to_csv(os.path.join('.', csv_name))
        generate_dataset_json(output_folder=dataset_dir,
                              channel_names={0: "CT"},
                              labels=label_key,
                              num_training_cases=len(train_files),
                              dataset_name=f"Dataset{args.dataset_num}_TemporalBone", license='hands off!')
    else:
        print(f"{output_dir} already exists.")

if __name__ == '__main__':
    parser = FilestructureArgParser()
    args_ = parser.parse_args()
    main(args_)
    ## For original dataset:
    # usage: python3 create_nnunet_filestructure.py -i ../niftis/ -o ../temp_ading/ -p ./datasplit.pkl -d 101
