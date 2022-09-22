# Config File Description
## Datasets
### Sources
- Train Dataset is taken from `/hpc-datasets/cars/combined_cars/tfrecords/labels_scvt/stratified_split_448_resize_without_pad/train`
  - SCVT is a combined dataset comprising Stanford Cars, CompCars, VMMRDb and Traffic Images
  - Sized 448 x 448 from original image without aspect ratio preservation
  - Shuffled
  - 80% of shuffled dataset
  - Training batch size is set to 448
- Validation Dataset is taken from `/hpc-datasets/cars/combined_cars/tfrecords/labels_scvt/stratified_split_448_resize_without_pad/train`
  - Sized 448 x 448 from original image without aspect ratio preservation
  - 20% of shuffled dataset
  - Validation batch size is set to 768 to maximize GPU usage and quicken validation processing
### Augmentations
- Train Dataset is 
  - Various augmentations using `albumentations`
  - Normalized to ImageNet mean/var
- Validation Dataset is 
  - Normalized to ImageNet mean/var
## Model
- The image first enters Attention Network made out of the first few blocks of ResNet50 followed by Affine Network which is made out of 3 small multi-layer-perceptrons (MLP) 
- The output from Affine Network is used in Spatial Transform Affine to generate a cropped image with higher activation sites
- The cropped image is fed into TResNet-M with pre-trained ImageNet weights which is used as feature extractor
- Feature extractor is fully connected to Global-K-Max-Pooling followed by a Dense, Batch Normalization and Dense layer
- The output of the above is fed through a L2 Normalized Dense layer for classification 
## Loss Function
- [Li-ArcFace Loss](https://arxiv.org/abs/1907.12256)
  - 40.0 scale, no margin
## Metrics
- Sparse Categorical Accuracy
## Optimizer
- [AdamW](https://arxiv.org/pdf/1711.05101.pdf)
## Train Loop Configuration
- Attention Net and Classifier weights are ImageNet pre-trained
- 40 epochs
### Callbacks
#### Learning Rate Scheduler
- [Cosine Decay with Restarts](https://arxiv.org/pdf/1608.03983.pdf)
#### KTF Variable Scheduler
- ktf.train.callbacks.VarScheduler
  - Used to schedule the AdamW `weight_decay` params. Implementation of [AdamW in tfa](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW) assumes constant learning rate, so we have to do it manually
#### Model Checkpoint
- Save best validation model
#### CSV Logger
- To log training progress
