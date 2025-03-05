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

## From the nnUNet README:
### Automatically determine the best configuration
Once the desired configurations were trained (full cross-validation) you can tell nnU-Net to automatically identify 
the best combination for you:

```commandline
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 
```

`CONFIGURATIONS` hereby is the list of configurations you would like to explore. Per default, ensembling is enabled 
meaning that nnU-Net will generate all possible combinations of ensembles (2 configurations per ensemble). This requires 
the .npz files containing the predicted probabilities of the validation set to be present (use `nnUNetv2_train` with 
`--npz` flag, see above). You can disable ensembling by setting the `--disable_ensembling` flag.

See `nnUNetv2_find_best_configuration -h` for more options.

nnUNetv2_find_best_configuration will also automatically determine the postprocessing that should be used. 
Postprocessing in nnU-Net only considers the removal of all but the largest component in the prediction (once for 
foreground vs background and once for each label/region).

Once completed, the command will print to your console exactly what commands you need to run to make predictions. It 
will also create two files in the `nnUNet_results/DATASET_NAME` folder for you to inspect: 
- `inference_instructions.txt` again contains the exact commands you need to use for predictions
- `inference_information.json` can be inspected to see the performance of all configurations and ensembles, as well 
as the effect of the postprocessing plus some debug information. 

### Run inference
Remember that the data located in the input folder must have the file endings as the dataset you trained the model on 
and must adhere to the nnU-Net naming scheme for image files (see [dataset format](dataset_format.md) and 
[inference data format](dataset_format_inference.md)!)

`nnUNetv2_find_best_configuration` (see above) will print a string to the terminal with the inference commands you need to use.
The easiest way to run inference is to simply use these commands.

If you wish to manually specify the configuration(s) used for inference, use the following commands:

#### Run prediction
For each of the desired configurations, run:
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

Only specify `--save_probabilities` if you intend to use ensembling. `--save_probabilities` will make the command save the predicted
probabilities alongside of the predicted segmentation masks requiring a lot of disk space.

Please select a separate `OUTPUT_FOLDER` for each configuration!

Note that per default, inference will be done with all 5 folds from the cross-validation as an ensemble. We very 
strongly recommend you use all 5 folds. Thus, all 5 folds must have been trained prior to running inference. 

If you wish to make predictions with a single model, train the `all` fold and specify it in `nnUNetv2_predict`
with `-f all`

#### Ensembling multiple configurations
If you wish to ensemble multiple predictions (typically form different configurations), you can do so with the following command:
```bash
nnUNetv2_ensemble -i FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -np NUM_PROCESSES
```

You can specify an arbitrary number of folders, but remember that each folder needs to contain npz files that were
generated by `nnUNetv2_predict`. Again, `nnUNetv2_ensemble -h` will tell you more about additional options.

#### Apply postprocessing
Finally, apply the previously determined postprocessing to the (ensembled) predictions: 

```commandline
nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
```

`nnUNetv2_find_best_configuration` (or its generated `inference_instructions.txt` file) will tell you where to find 
the postprocessing file. If not you can just look for it in your results folder (it's creatively named 
`postprocessing.pkl`). If your source folder is from an ensemble, you also need to specify a `-plans_json` file and 
a `-dataset_json` file that should be used (for single configuration predictions these are automatically copied 
from the respective training). You can pick these files from any of the ensemble members.


## How to run inference with pretrained models
See [here](run_inference_with_pretrained_models.md)

## How to Deploy and Run Inference with YOUR Pretrained Models
To facilitate the use of pretrained models on a different computer for inference purposes, follow these streamlined steps:
1. Exporting the Model: Utilize the `nnUNetv2_export_model_to_zip` function to package your trained model into a .zip file. This file will contain all necessary model files.
2. Transferring the Model: Transfer the .zip file to the target computer where inference will be performed.
3. Importing the Model: On the new PC, use the `nnUNetv2_install_pretrained_model_from_zip` to load the pretrained model from the .zip file.
Please note that both computers must have nnU-Net installed along with all its dependencies to ensure compatibility and functionality of the model.

