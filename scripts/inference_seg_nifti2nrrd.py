import slicerio
import nrrd
import csv
import pandas as pd
import argparse
import sys
import os
import glob

def read_colortable(colortable_filename):
    with open(colortable_filename, mode='r') as infile:
        reader = csv.reader(infile)
        colortable_header = next(reader)
        colortable = {i:(rows[0],rows[1][1:-1].split(',')) for i,rows in enumerate(reader)}
    return colortable

def read_mapping(mapping_filename):
    with open(mapping_filename, mode='r') as infile:
        reader = csv.reader(infile)
        mapping_header = next(reader)
        mapping = {rows[0]:rows[1] for rows in reader}
    return mapping

def segnifti_to_nrrd(input_filename, output_filename, colortable):
    segmentation_info = slicerio.read_segmentation(input_filename)
    number_of_segments = len(segmentation_info['segments'])
    print(f"Number of segments: {number_of_segments}")
    segments = segmentation_info['segments']
    for i, segment in enumerate(segments):
        segment['name'] = colortable[i][0]
        segment['color'] = colortable[i][1]

    slicerio.write_segmentation(output_filename, segmentation_info)

def main(argv):
    input_folder = argv[0]
    output_folder = argv[1]
    mapping_filename = argv[2]
    colortable_filename = argv[3]
    datasets = [os.path.basename(f).split('.nii.gz')[0] for f in glob.glob(os.path.join(input_folder, 'jhu***.nii.gz'))]
    mapping = read_mapping(mapping_filename)
    colortable = read_colortable(colortable_filename)

    for dataset in datasets:
        print(f"Converting {dataset} NIFTI to {mapping[dataset]} NRRD")
        input_filename = os.path.join(input_folder, dataset + '.nii.gz')
        output_filename = os.path.join(output_folder, 'Segmentation_' + mapping[dataset] + '.seg.nrrd')
        segnifti_to_nrrd(input_filename, output_filename, colortable)    

if __name__ == '__main__':
    main(sys.argv[1:])

# python3 inference_seg_nifti2nrrd.py /home/andyding/tbone-seg-nnunetv2/00_nnUNetv2_baseline_retrain/nnUNet_trained_models/Dataset101_TemporalBone/nnUNetTrainer_300epochs__nnUNetPlans__3d_fullres/inference_results/reinfer /home/andyding/tbone-seg-nnunetv2/00_nnUNetv2_baseline_retrain/nnUNet_trained_models/Dataset101_TemporalBone/nnUNetTrainer_300epochs__nnUNetPlans__3d_fullres/inference_results/renamed /home/andyding/tbone-seg-nnunetv2/00_nnUNetv2_baseline_retrain_total_mapping.csv /home/andyding/tbone-seg-nnunetv2/scripts/tbone_colortable.csv