'''
Code borrowed from:
https://github.com/rachellea/ct-volume-preprocessing/blob/master/preprocess_volumes.py#L388
'''

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import time


def resample_volume(ctvol_path, out_spacing, out_path, verbose = True):
        """Resample volume represented as a Path objectto nifti file <ctvol> to desired
        spacing and return <desired_spacing>. Incudes IO operations.
        """
        start_time = time.time()
        assert isinstance(ctvol_path, Path)

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
        resample.SetInterpolator(sitk.sitkBSpline)
        resampled_ctvol = resample.Execute(ctvol_itk)
        if verbose: print('actual out shape in sitk:',resampled_ctvol.GetSize()) #e.g. [388, 469, 469]
        assert [x for x in resampled_ctvol.GetSize()]==out_shape, 'Error in resample_volume: incorrect sitk resampling shape obtained' #make sure we got the shape we wanted
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(out_path))
        writer.Execute(resampled_ctvol)

        end_time = time.time()
        runtime = end_time - start_time
        if verbose: print('Total runtime for resampling:',runtime)

  
def main():
    ctvol_path = Path('/home/andyding/tbone-seg-nnunetv2/00_nnUNetv2_baseline_retrain/nnUNet_raw/Dataset101_TemporalBone/test_cadaver_tbone_1/jhu_060_0000_cropped_unresampled.nii.gz')
    out_path = Path('/home/andyding/tbone-seg-nnunetv2/00_nnUNetv2_baseline_retrain/nnUNet_raw/Dataset101_TemporalBone/test_cadaver_tbone_1/jhu_060_0000.nii.gz')
    desired_spacing = [0.09797599911689758, 0.09796378761529922, 0.09797599911689758]
    resample_volume(ctvol_path, desired_spacing, out_path, verbose = True)

if __name__ == '__main__':
    main()
