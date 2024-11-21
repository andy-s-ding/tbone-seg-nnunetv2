"""
compute_accuracy_metrics_nnunet.py

Compute dice scores and Hausdorff distances for segment ids
Ground truth and predicted segmentation files should be .nii.gz

"""
import os 
import sys 
import argparse 
import numpy as np
import pandas as pd
import nibabel as nib
import surface_distance
from surface_distance.metrics import *
import glob
import fnmatch

SEG_NAMES = {
    0: "Background",
    1: "Bone",
    2: "Malleus",
    3: "Incus",
    4: "Stapes",
    5: "Bony_Labyrinth",
    6: "Vestibular_Nerve",
    7: "Superior_Vestibular_Nerve",
    8: "Inferior_Vestibular_Nerve",
    9: "Cochlear_Nerve",
    10: "Facial_Nerve",
    11: "Chorda_Tympani",
    12: "ICA",
    13: "Sigmoid_Sinus",
    14: "Dura",
    15: "Vestibular_Aqueduct",
    16: "Mandible",
    17: "EAC",
}

def parse_command_line(args):
    print('parsing command line')
    parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
    parser.add_argument('--pred',
                        action="store",
                        type=str,
                        help="Specify a prediction folder"
                        )
    parser.add_argument('--gt',
                        action="store",
                        type=str,
                        help="Specify a prediction folder"
                        )
    parser.add_argument('--folds',
                        type=int,
                        nargs='+',
                        help="Validation fold folders to evaluate")
    parser.add_argument('--ids',
                        type=int,
                        nargs='+',
                        help="Segment indices (1-indexed) to calculate accuracy metrics")

    
    args = vars(parser.parse_args())
    return args

def calculate_accuracy_metrics(pred_dir, gt_dir, target_list, ids,
                               dice_dict=dict(), hausdorff_dict=dict(),
                               seg_names=SEG_NAMES, return_output=False):
    try: dice_dict['Target']
    except:
        dice_dict['Target'] = []
        for i in ids: dice_dict[seg_names[i]] = []
    try: hausdorff_dict['Target']
    except:
        hausdorff_dict['Target'] = []
        for i in ids: hausdorff_dict[seg_names[i]] = []

    for target in target_list:
        print('-- Evaluating %s'%(target))
        pred_seg_path = os.path.join(pred_dir, "{}.nii.gz".format(target))
        gt_seg_path = os.path.join(gt_dir, "{}.nii.gz".format(target))

        pred_seg = np.array(nib.load(pred_seg_path).dataobj)
        gt_seg = np.array(nib.load(gt_seg_path).dataobj)
        spacing = nib.load(gt_seg_path).header.get_zooms()

        pred_one_hot = np.moveaxis((np.arange(pred_seg.max()+1) == pred_seg[...,None]), -1, 0)
        gt_one_hot = np.moveaxis((np.arange(gt_seg.max()+1) == gt_seg[...,None]), -1, 0)

        print(pred_one_hot.shape)
        print(gt_one_hot.shape)

        dice_dict['Target'].append(target)
        hausdorff_dict['Target'].append(target)

        for i in ids:
            print('---- Computing metrics for segment: %s'%(seg_names[i]))
            dice_coeff = compute_dice_coefficient(gt_one_hot[i], pred_one_hot[i])
            surface_distances = compute_surface_distances(gt_one_hot[i], pred_one_hot[i], spacing)
            mod_hausdorff_distance = max(compute_average_surface_distance(surface_distances))

            dice_dict[seg_names[i]].append(dice_coeff)
            hausdorff_dict[seg_names[i]].append(mod_hausdorff_distance)

    if return_output:
        return dice_dict, hausdorff_dict

def main(): 
    args = parse_command_line(sys.argv)
    gt_dir = args['gt']
    pred_dir = args['pred']
    folds = args['folds']
    ids = args['ids']
    
    # Initialize metric dictionaries
    dice_dict, hausdorff_dict = dict(), dict()
    dice_dict['Target'], hausdorff_dict['Target'] = [], []
    if not ids: ids = list(range(1,len(SEG_NAMES))) # All structures
    for i in ids: dice_dict[SEG_NAMES[i]], hausdorff_dict[SEG_NAMES[i]] = [], []
        
    if folds:
        for fold in folds:
            print("Fold {}".format(fold))
            fold_dir = os.path.join(pred_dir, "fold_{}".format(fold), "validation")
            target_list = [target.split('.nii.gz')[0] for target in os.listdir(fold_dir) if fnmatch.fnmatch(target, '*.nii.gz')]
            calculate_accuracy_metrics(fold_dir, gt_dir, target_list, ids, dice_dict, hausdorff_dict, SEG_NAMES)
    else:
        target_list = [target.split('.nii.gz')[0] for target in os.listdir(pred_dir) if fnmatch.fnmatch(target, '*.nii.gz')]
        calculate_accuracy_metrics(pred_dir, gt_dir, target_list, ids, dice_dict, hausdorff_dict, SEG_NAMES)

    dice_df = pd.DataFrame.from_dict(dice_dict)
    hausdorff_df = pd.DataFrame.from_dict(hausdorff_dict)

    print('Dice Scores')
    print(dice_df)

    print('Modified Hausdorff Distances')
    print(hausdorff_df)

    dice_path = os.path.join(pred_dir, 'dice ' + '-'.join(str(i) for i in ids) + '.csv')
    hausdorff_path = os.path.join(pred_dir, 'hausdorff ' + '-'.join(str(i) for i in ids) + '.csv')

    dice_df.to_csv(dice_path)
    hausdorff_df.to_csv(hausdorff_path)

    return 

if __name__ == "__main__":
    main()
# python3 compute_accuracy_metrics_nnunet.py --pred /home/andyding/tbone-seg-nnunetv2/01_nnUNetv2_tbone/nnUNet_trained_models/Dataset101_TemporalBone/nnUNetTrainer_300epochs__nnUNetPlans__3d_fullres/ --folds 0 1 2 3 4 --gt /home/andyding/tbone-seg-nnunetv2/01_nnUNetv2_tbone/nnUNet_raw/Dataset101_TemporalBone/labelsTr/