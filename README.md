# tbone-seg-nnunetv2

**Important:** This pipeline has been validated for Ubuntu 22.04 LTS.
Instructions for Temporal Bone Dataset Use:

## Step 0: Clone repo
```
git clone https://github.com/andy-s-ding/tbone-seg-nnunetv2
```

## Step 1: Create datasplit for training/testing. Validation will automatically be chosen
Navigate to the `scripts` folder:
```
cd <path to github>/tbone-seg-nnunet/scripts/
```

The datasplit file will be a `.pkl` file that will be referenced when creating the final file structure for nnUNet training.

For the general (default) dataset without deformation field SSM generation, this is done by:
```
python create_datasplit.py
```
For the deformation field SSM-generated dataset, this is done by:
```
python create_generated_datasplit.py
```
Note that in order to create the SSM-generated datasplit, the general `datasplit.pkl` file needs to exist first. This is because the generated datasplit uses the same test set as the general split.

## Step 2: Create file structure required for nnUNet
Create a base directory `tbone-seg-nnunet/<BASE_DIR>` that will serve as the root directory for the nnUNet training file structure.

In the `scripts/` folder, run `create_nnunet_filestructure.py` to copy training and test data over based on the datasplit `.pkl` generated in Step 3.

For the general temporal bone dataset:
```
python create_nnunet_filestructure.py --dataset original --original_dataset_dir <registered original images> --output_dir <BASE_DIR> --pickle_path ./datasplit.pkl, --task_num <task num>
```
For the SSM generated datasplit, this is done by:
```
python create_nnunet_filestructure.py --input_dir <original images dir> --output_dir <BASE_DIR> --pickle ./datasplit_generated.pkl --dataset_num <dataset_num>
```

## Step 3: Setup bashrc
Edit your `~/.bashrc` file with `gedit ~/.bashrc` or `nano ~/.bashrc`. At the end of the file, add the following lines:
```
export nnUNet_raw="<ABSOLUTE PATH TO BASE_DIR>/nnUNet_raw" 
export nnUNet_preprocessed="<ABSOLUTE PATH TO BASE_DIR>/nnUNet_preprocessed" 
export nnUNet_results="<ABSOLUTE PATH TO BASE_DIR>/nnUNet_trained_models"
```
After updating this you will need to source your `~/.bashrc` file:
```
source ~/.bashrc
```

## Step 4: Verify and preprocess data
Run the nnUNet preprocessing script:
```
nnUNetv2_plan_and_preprocess -d <dataset_num> --verify_dataset_integrity
```
Potential Error: You may need to edit the dataset.json file so that the labels are sequential. If you have at least 10 labels, then labels `10, 11, 12,...` will be arranged before labels `2, 3, 4, ...`. Doing this in a text editor is completely fine!

### Step 4a: Setting up Training with Distace Maps
Extra steps are needeed to set up training using distance map-weighted loss functions. First, the distance maps must be pre-computed from the preprocessed data. In the `scripts` folder:
```
python3 compute_distance_maps.py --input_dir <BASE_DIR>/nnUNet_preprocessed/Dataset<datset_num>_TemporalBone/nnUNetPlans_3d_fullres/ --multi_class
```
This will save distance maps as `jhu_<id_num>_dist.npy` files for training, similar to how labels are saved as `jhu_<id_num>_seg.npy` files. Using the `--nifti` option will save `.nii.gz` files for you to view them in Slicer or another medical image analysis software.

Running this script requires the [Multi-Label Anisotropic 3D Euclidean Distance Transform (MLAEDT-3D)](https://github.com/seung-lab/euclidean-distance-transform-3d) package, which computes multi-class distance transforms in O(N) time rather than O(N<sup>3</sup>) time.

## Step 5: Begin Training
For vanilla training on a 3D nnUNet, run:
```
nnUNetv2_train Dataset<dataset_num>_TemporalBone 3d_fullres nnUNetTrainer <fold_num>
```
`<fold_num>` refers to the fold number (0 to 4) for cross-validation. If `<fold_num>` is set to `all` then all of the data will be used for training.

Variants of the `nnUNetTrainer` class can be made and saved in `nnUNet/nnunetv2/training/nnUNetTrainer/variants/`. Refer to other variants for examples. Distance-map based training uses the `nnUNet/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerDistDiceLoss.py` variant. Training with a custom `nnUNetTrainer` variant can then be run as:

```
nnUNetv2_train Dataset<dataset_num>_TemporalBone 3d_fullres <nnUNetTrainer Variant Name> <fold_num>
```
Multiple variants can be trained on the same dataset.

