import slicerio
import nrrd
import csv
import pandas as pd
import argparse
import sys
import os
import glob
from collections import OrderedDict

SEG_NAMES = {
    1: "Bone",
    2: "Malleus",
    3: "Incus",
    4: "Stapes",
    5: "Bony Labyrinth",
    6: "Vestibular Nerve",
    7: "Superior Vestibular Nerve",
    8: "Inferior Vestibular Nerve",
    9: "Cochlear Nerve",
    10: "Facial Nerve",
    11: "Chorda Tympani",
    12: "ICA",
    13: "Sigmoid Sinus",
    14: "Dura",
    15: "Vestibular Aqueduct",
    16: "Mandible",
    17: "EAC",
}

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
    if number_of_segments != len(colortable):
        print(f"Some segments are blank")
    segmentation_info['segments'] = [OrderedDict([('labelValue', i+1),
                                                  ('name', colortable[i][0]),
                                                  ('color', colortable[i][1])]) 
                                                  for i in range(len(colortable))]

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

# python3 inference_seg_nifti2nrrd.py /home/andyding/Desktop /home/andyding/Desktop/008 /home/andyding/tbone-seg-nnunetv2/00_nnUNetv2_baseline_retrain_total_mapping.csv /home/andyding/tbone-seg-nnunetv2/scripts/tbone_colortable.csv