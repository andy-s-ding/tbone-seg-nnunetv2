"""
create_datasplit.py

Creates datasplit based on subjects *NOT* files
arg1: Segmentation folder
arg2 (optional): .pkl output name
"""

import os
import glob
import random
import pickle
import sys
from pathlib import Path

def filename(file):
    return os.path.basename(file).split('.')[0]

def main(argv):
    random.seed(0)
    data_path = argv[0]
    try: save_file = argv[1]
    except: save_file = 'datasplit.pkl'

    file_list = glob.glob(os.path.join(data_path, "Segmentation_*.nii.gz"))
    # File IDs including left/right distinctions
    file_ids = [filename(f).split("Segmentation_")[-1] for f in file_list]
    num_files = len(file_ids)
    # Non-duplicate number of ears (subjects)
    subject_ids = list(set([f.split('_')[-1] for f in file_ids]))
    num_subjects = len(set(subject_ids))

    # Dataset Split (cumulative sum):
    num_train = int(round(num_subjects*0.7))
    num_test = num_subjects - num_train
    print("Number of training subjects: ", num_train, "\nNumber of testing subjects:", num_test, "\nTotal:", num_subjects)
    assert(num_train + num_test == num_subjects)

    random.shuffle(subject_ids)
    train_subjects = random.sample(subject_ids, num_train)
    test_subjects = [subject for subject in subject_ids if subject not in train_subjects]

    train_files = [f for f in file_ids if any(subject in f for subject in train_subjects)]
    test_files = [f for f in file_ids if any(subject in f for subject in test_subjects)]

    split = {"Train": train_files, "Test": test_files}
    
    print(split)
    
    with open(save_file, 'wb') as pickle_out:
        pickle.dump(split, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == '__main__':
    main(sys.argv[1:])
    # usage: python3 create_datasplit.py ../niftis/