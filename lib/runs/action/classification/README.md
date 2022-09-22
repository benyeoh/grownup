# Config File for Experimental Run of Zero Shot Action Classification
## References
### Paper
https://arxiv.org/pdf/2003.01455.pdf
### Code
https://github.com/bbrattoli/ZeroShotVideoClassification
## Results
| Author \| Our Config | Zero Shot Sparse Top 1 Categorical Accuracy for UCF101 | Zero Shot Sparse Top 1 Categorical Accuracy for HMDB51 | Reference | Note |
|---|---|---|---|---|
| (Author)<br><br>num_epoch: 50<br>train_dataset: Kinetics (664 classes)<br>val_dataset: UCF101 & HMDB51<br>num_GPU: 4<br>total_batch_size: 88 | 37.6 | 26.9 | PyTorch Checkpoint: https://drive.google.com/file/d/1U35017nrmwh0_RMbaDxr-CxUurS6pHfU/view?usp=sharing<br>- Look under key `opt`<br>```Namespace(beta=0.0, bs=88, class_overlap=0.01, class_total=-1, clip_len=16, dataset='kinetics7002both', device=device(type='cuda'), features=None, fixconvs=False, kernels=32, loss='mse', lr=1e-05, multiple_clips=False, n_classes=664, n_clips=1, n_epochs=50, network='r2plus1d_18', nonlinear=False, progressbar=False, sample_skip=-1, save_path='/workplace/data/motion_efs/home/biagib/ZeroShot/reduced_training/', savename='/workplace/data/motion_efs/home/biagib/ZeroShot/reduced_training//kinetics7002both/MSE_NCLIPS1_CLIP16_LR0.000100_r2plus1d_18_BS88_BETA0.000000_CLASSOVERLAP0.01', size=112, split=-1, train_samples=-1, use_lstm=False, weights='none')```<br>- Cross reference with code and paper | - PyTorch weights cannot be loaded for verification purposes as there is a mismatch of layer weights<br>- - Suspect that there is an additional dense layer of size num_classes after Resnet18 (2D+1) feature extraction before output is passed out for averaging |
| (Author)<br><br>num_epoch: 70<br>pre-trained weights: trained on SUN<br><br>train_dataset: Kinetics (678 classes)<br>val_dataset: UCF101 & HMDB51<br>num_GPU: 8<br>total_batch_size: 176 | 39.8 | Not provided in paper | PyTorch Checkpoint: https://drive.google.com/file/d/1bzy_qfx7Jlfj8CUiHltRN4YnT4GEG-ly/view?usp=sharing<br>- Look under key `opt`<br>```Namespace(bs=176, class_overlap=0.05, class_total=-1, clip_len=16, dataset='kinetics7002both', device=device(type='cuda'), evaluate=False, fixconvs=False, kernels=64, loss='mse', lr=1e-05, multiple_clips=False, n_classes=678, n_clips=1, n_epochs=70, network='r2plus1d_18', pretrained=True, progressbar=False, save_path='/workplace/data/motion_efs/home/biagib/ZeroShot/reduced_training/sun_pretrained_samples/', savename='/workplace/data/motion_efs/home/biagib/ZeroShot/reduced_training/sun_pretrained_samples//kinetics7002both/MSE_NCLIPS1_CLIP16_LR0.000100_r2plus1d_18_BS176_CLASSOVERLAP0.05', size=112, split=-1, train_samples=-1, weights='/workplace/data/motion_efs/home/biagib/ZeroShot/reduced_training/classes_stillimages_uniformsampling/kinetics2both_images/MSE_NCLIPS1_CLIP16_LR0.000100_r2plus1d_18_BS176_CLASSOVERLAP0.05_NCLASS0/checkpoint.pth.tar')```<br>- Cross reference with code and paper | - Training was done on 678 classes on Kinetics rather than 664 classes |
| (Ours)<br><br>num_epoch: 70<br>train_dataset: Kinetics (664 classes)<br>val_dataset: UCF101<br>num_GPU: 4<br>total_batch_size: 88 | Epoch 46: 38.30<br><br>Epoch 65: 38.438 | Epoch 46: 21.53<br><br>Epoch 59: 22.47<br><br>Epoch 65: 22.12 | Training Log:<br>- ```/hpc-datasets/ck/ktf/multi_clip_video_resnet/logs/run1/run1_1_train.csv``` | Disclaimer:<br>- For UCF101:<br>- - Training was actually stopped at Epoch 18 and restarted again on Epoch 19<br>- - The training config provided is a summarised config putting the 2 stages of training to 1 config<br>- For HMDB51:<br>- - No separate training was done. <br>- - Weights were taken from training (where validation is UCF101) and validated on HMDB51 |
## Config File
### Location
```klassterfork/python/runs/action/classification/config_author_wo_sun_ucf101.hjson```
### Description
#### Dataset
##### Source
- Training dataset is Kinetics 700 (excluding 36 classes as mentioned in the paper)
  - Before conversion to TFRecord, frames have been resized such that the shorter side is of length 128
  - During reading of TFRecords, for each video, a clip of 16 frames is randomly selected and from these, a random crop of 112x112 is taken
- Validation dataset is the whole of UCF 101 or HMDB51
  - Before conversion to TFRecord, frames have been resized such that the shorter side is of length 128
  - During reading of TFRecords, for each video, 25 clips of 16 frames is randomly selected and from these, a central crop of 112x112 is taken
##### Augmentations
###### Training
- Random horizontal flip followed by frame standardization
###### Validation
- Frame standardization
#### Model
- Base model is Resnet18(2D+1) with global average pooling
- During inference, each clip is inferred on base model and the resultant feature vectors and averaged and normalized. Refer to code for more details.
#### Loss
- MSE
#### Metrics
- Sparse Categorical Accuracy (Top 1 and Top 5)
#### Optimizer
- Adam
##### Learning rate schedule:
##### UCF101
- Epoch 1-40: 1e-3
- Epoch 41-61: 1e-4
- Epoch 62-70: 1e-5
#### Training Settings
- 70 Epochs
