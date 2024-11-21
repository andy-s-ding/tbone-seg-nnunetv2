'''
Code borrowed from:
https://github.com/rachellea/ct-volume-preprocessing/blob/master/preprocess_volumes.py#L388
'''

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import time
import argparse
import sys
import os

def parse_command_line(args):
    '''
    '''
    print('parsing command line')

    parser = argparse.ArgumentParser(description='NIFTI Resampler')
    parser.add_argument('-i', '--input',
                        action="store",
                        type=str,
                        default=None,
                        help="Input NIFTI"
                        )
    parser.add_argument('-o', '--output',
                        action="store",
                        type=str,
                        default=None,
                        help="Output NIFTI"
                        )
    parser.add_argument('-r', '--reference',
                        action="store",
                        type=str,
                        default=None,
                        help="Reference NIFTI"
                        )
    parser.add_argument('-s', '--spacing',
                        type=float,
                        nargs='+',
                        default=[0.09797599911689758, 0.09796378761529922, 0.09797599911689758],
                        help="Manual spacing request"
                        )
    parser.add_argument('--overwrite',
                        action="store_true",
                        help="True if overwriting input volume"
                        )
    parser.add_argument('v', '--verbose',
                        action="store_true",
                        help="True if verbose"
                        )

    args = vars(parser.parse_args())
    print(args)
    return args


def resample_volume(ctvol_path, out_spacing, out_path, labelmap=False, verbose=True):
        """Resample volume  nifti file <ctvol> to desired spacing.
        """
        start_time = time.time()

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(str(ctvol_path))
        ctvol_itk = reader.Execute();

        original_size = ctvol_itk.GetSize()
        original_spacing = ctvol_itk.GetSpacing()

        if verbose: print('ctvol before resampling',original_size) #e.g. [512, 512, 518]
        if verbose: print('ctvol original spacing:',original_spacing) #e.g. [0.6, 0.732421875, 0.732421875]
        
        #Calculate out shape:
        #Relationship: (origshape x origspacing) = (outshape x outspacing)
        #in other words, we want to be sure we are still representing
        #the same real-world lengths in each direction.
        out_shape = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
        if verbose: print('desired out shape:',out_shape) #e.g. [469, 469, 388]
      
        #Perform resampling:
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_shape)
        resample.SetOutputDirection(ctvol_itk.GetDirection())
        resample.SetOutputOrigin(ctvol_itk.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(ctvol_itk.GetPixelIDValue())
        if labelmap: resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else: resample.SetInterpolator(sitk.sitkBSpline)
        resampled_ctvol = resample.Execute(ctvol_itk)
        if verbose: print('actual out shape in sitk:',resampled_ctvol.GetSize()) #e.g. [388, 469, 469]
        assert [x for x in resampled_ctvol.GetSize()]==out_shape, 'Error in resample_volume: incorrect sitk resampling shape obtained' #make sure we got the shape we wanted
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(out_path))
        writer.Execute(resampled_ctvol)

        end_time = time.time()
        runtime = end_time - start_time
        if verbose: print('Total runtime for resampling: ', runtime)

def resample_volume_from_reference(ctvol_path, ctvol_ref_path, out_path, labelmap=False, verbose=True):
        """Resample volume nifti file <ctvol> to match reference nifti file <ctvol_ref_path>
        """
        start_time = time.time()

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(str(ctvol_path))
        ctvol_itk = reader.Execute();

        original_size = ctvol_itk.GetSize()
        original_spacing = ctvol_itk.GetSpacing()

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(str(ctvol_ref_path))
        ctvol_ref_itk = reader.Execute();
        out_spacing = ctvol_ref_itk.GetSpacing()

        if verbose: print('ctvol before resampling',original_size) #e.g. [512, 512, 518]
        if verbose: print('ctvol original spacing:',original_spacing) #e.g. [0.6, 0.732421875, 0.732421875]
        
        #Calculate out shape:
        #Relationship: (origshape x origspacing) = (outshape x outspacing)
        #in other words, we want to be sure we are still representing
        #the same real-world lengths in each direction.
        out_shape = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
        if verbose: print('desired out shape:',out_shape) #e.g. [469, 469, 388]

        #Perform resampling:
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_shape)
        resample.SetOutputDirection(ctvol_itk.GetDirection())
        resample.SetOutputOrigin(ctvol_itk.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(ctvol_itk.GetPixelIDValue())
        if labelmap: resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else: resample.SetInterpolator(sitk.sitkBSpline)
        resampled_ctvol = resample.Execute(ctvol_itk)
        if verbose: print('actual out shape in sitk:',resampled_ctvol.GetSize()) #e.g. [388, 469, 469]
        assert [x for x in resampled_ctvol.GetSize()]==out_shape, 'Error in resample_volume: incorrect sitk resampling shape obtained' #make sure we got the shape we wanted
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(out_path))
        writer.Execute(resampled_ctvol)

        end_time = time.time()
        runtime = end_time - start_time
        if verbose: print('Total runtime for resampling: ', runtime)

  
def main(argv):
    args = parse_command_line(sys.argv)
    ctvol_path = args['input']
    ctvol_ref_path = args['reference']
    out_path = args['output']
    if out_path is None:
        if args['overwrite']: out_path = ctvol_path
        else: out_path = ctvol_path.split('.nii.gz')[0]+'_resampled.nii.gz'
    spacing = args['spacing']

    if ctvol_ref_path:
        resample_volume_from_reference(ctvol_path, ctvol_ref_path, out_path, verbose=args['verbose'])
    else: resample_volume(ctvol_path, spacing, out_path, verbose=args['verbose'])

if __name__ == '__main__':
    main()