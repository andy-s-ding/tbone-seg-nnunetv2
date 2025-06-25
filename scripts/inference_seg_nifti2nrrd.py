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

def parse_command_line(args):
    print('parsing command line')
    parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
    parser.add_argument('-i', '--input',
                        action="store",
                        type=str,
                        help="Specify an input folder",
                        required=True
                        )
    parser.add_argument('-o', '--output',
                        action="store",
                        type=str,
                        help="Specify an output folder",
                        required=True
                        )
    parser.add_argument('-m', '--map',
                        action="store",
                        type=str,
                        help="Specify a mapping .csv",
                        default=None
                        )
    parser.add_argument('-c', '--color',
                        action="store",
                        type=str,
                        help="Specify a colortable .csv",
                        required=True
                        )
    
    args = vars(parser.parse_args())
    return args

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

def main():
    args = parse_command_line(sys.argv)
    input_folder = args['input']
    output_folder = args['output']
    os.makedirs(output_folder, exist_ok=True)
    mapping_filename = args['map']
    colortable_filename = args['color']
    datasets = [os.path.basename(f).split('.nii.gz')[0] for f in glob.glob(os.path.join(input_folder, 'jhu***.nii.gz'))]
    mapping = read_mapping(mapping_filename) if mapping_filename is not None else None
    colortable = read_colortable(colortable_filename)
    
    print(datasets)
    for dataset in datasets:
        print(f"Converting {dataset} NIFTI to NRRD")
        input_filename = os.path.join(input_folder, dataset + '.nii.gz')
        try:
            print(f"Renaming {dataset} to {mapping[dataset]}")
            output_filename = os.path.join(output_folder, 'Segmentation_' + mapping[dataset] + '.seg.nrrd')
        except:
            print(f"Mapping of {dataset} not found. File name will be {dataset}")
            output_filename = os.path.join(output_folder, 'Segmentation_' + dataset + '.seg.nrrd')
        segnifti_to_nrrd(input_filename, output_filename, colortable)    

if __name__ == "__main__":
    main()

# python3 inference_seg_nifti2nrrd.py /home/andyding/Desktop /home/andyding/Desktop/008 /home/andyding/tbone-seg-nnunetv2/00_nnUNetv2_baseline_retrain_total_mapping.csv scripts/tbone_colortable_separated_sinus_dura.csv