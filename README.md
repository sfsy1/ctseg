# CT Segmentation
 All kinds of models and scripts for data processing and visualizations.


## Environment
Python3.12 is recommended. Libraries are managed by https://python-poetry.org/

After installing poetry, run this to start poetry venv:
```commandline
poetry install
```

Create a jupyter kernel from venv:
```commandline
poetry run python -m ipykernel install --user --name ctseg-py3.12
```

To edit or remove kernels:
```commandline
jupyter kernelspec list
jupyter kernelspec remove KERNEL_NAME
```
## 2D Segmentation with BBox Prompt
* Main folder: `2d_seg_and_video`
* Training: `2d_seg_and_video/2d_box_seg.ipynb`
* Processing logic mostly in: `2d_seg_and_video/dataset.py`
  * Cropping and resizing
  * Adding bbox prompt

![img.png](readme/2d_bbox_prompt_seg.png)
* Green: true positive
* Red: false positive
* Purple: false negative

### Dataset
DeepLesion 3D subset of the ULS23 dataset
* 743 3D lesion segmentations
* 4538 2D slices
* train/val split done by lesions to avoid data leakage


## Video Generation
### AutoPET
[Dataset](https://uppsala.app.box.com/folder/286456299982?s=t33kcqjifp0q23fv2zf0i58sz8njxcd7)
that we are labeling and reviewing (2024-2025).

Script: `notebooks/ct_video.ipynb`

## ULS23
* Download all 6 .zip parts files from https://github.com/DIAGNijmegen/ULS23/
* Unzip all 6 parts such that they are in the same folder 
  * The folder should contain fragmented zip files (.z01, .z02, ...) for all 6 parts
* Unzip the fragmented zip files

```commandline
sudo apt update
sudo apt install p7zip-full

7z x ULS23_Part1.zip
7z x ULS23_Part2.zip
...
```

Labels in git repo - merge the label folder with the existing `ULS23` data folder.

## 3D segmentation
* Main folder: `3d_seg`

### CT-FM Seg Model
* Project page: https://aim.hms.harvard.edu/ct-fm
* Model: https://huggingface.co/project-lighter/whole_body_segmentation 

#### Out of the box results on AutoPET
![img.png](readme/ct_fm_sample.png)

#### Relevant files
* `patching.ipynb` Scripts to split existing datasets into patches
* `ct-fm.ipynb` Scripts for testing and training ct-fm seg model
* `lesion3D` Dataset folder in the Kingston SSD

###  nnUNetv2
[nnUNet repo](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md) 
All commands below should be run in the nnUNet project directory.

#### Relevant files
* `nnunet_data_proc.ipynb` Scripts to generate raw data
* `nnUNet_raw` Dataset folder in the Kingston SSD

#### Setting up
Set env variables e.g. by creating an `env.sh` file:
```bash
export nnUNet_raw="path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed/"
export nnUNet_results="/path/to/nnUNet_results"
```
Apply them in terminal:
```bash
source env.sh
```
#### Preprocess
At 64GB memory, 2 processes `-np 2` is the max possible on this dataset. Proocessed data will be stored at the `nnUNet_preprocessed` path.
```bash
nnUNetv2_plan_and_preprocess -d 1 -np 2
```


#### Train
Train model dataset (1 in this example) on a specified cross-validation fold (0 in this example).
```bash
nnUNetv2_train 1 3d_fullres 0 --npz
```
Other training config are `2d`, `3d_lowres`, `3d_cascade_fullres`. The log and results will be stored at the `nnUNet_preprocessed` path.


##### Resume Training
Save the best and final checkpoints from previous run before running this:
```bash
nnUNetv2_train 1 3d_fullres all --npz -pretrained_weights path/to/checkpoint.pth
```

#### Adding extra channels
* CT channel is always 0000
* Seg masks from CT-FM seg model is 0001
* Boxes and other priors are 0002

Sample `dataset.json`
```
{
   "channel_names":{
      "0":"CT",
      "1": "noNorm"
   },
   "labels":{
      "background":0,
      "lesion":1
   },
   "numTraining":782,
   "file_ending":".nii.gz"
}
```
#### Data Generation
* `studies_ctfm_seg_mask.ipynb` generates the anatomical mask channels
* `studies_weak_labels.ipnyb` generates the box and mask channels

#### Data Splits
The default 5 folds are used. I used the same folds every dataset
by copying the `splits_final.json` file from the preprocessed folder of the first
dataset i.e. `nnUNet_preprocessed/Dataset001_3dlesion/`


## Visualization of 3D Outputs
* `3d_seg/visual.ipynb` and `3d_seg/visualize.py` contains code for visualizing 3D 
predictions for all the studies along with the CT and label masks.
* Visualization of val set can be found on the Kingston SSD `3d_val_visualization` folder


## Metrics & Results
Both voxel and lesion level metrics are calculated.
* `3d_seg/metrics.ipynb` contains code for calculating metrics
* `3d_seg/metrics/` contains the csv files with metrics for every lesion in val set
