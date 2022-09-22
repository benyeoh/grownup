# Config File Description
## Datasets
### Sources
- Train Dataset is taken from `hpc-datasets/cars/stanford_car/original_kaggle/tfrecord_448_resize_without_pad/train`
  - Sized 448 x 448 from original image without aspect ratio preservation
  - Shuffled
  - Training batch size is set to 64
- Validation Dataset is taken from `hpc-datasets/cars/stanford_car/original_kaggle/tfrecord_448_resize_without_pad/test`
  - Sized 448 x 448 from original image without aspect ratio preservation
  - Validation batch size is set to 512 to maximize GPU usage and quicken validation accuracy processing
### Augmentations
- Train Dataset is 
  - Various augmentations using `albumentations`
  - Normalized to ImageNet mean/var
- Validation Dataset is 
  - Normalized to ImageNet mean/var
## Model
- The image first enters Attention Network made out of the first few blocks of ResNet50 followed Affine Network which is made out of 3 small multi-layer-perceptrons (MLP) 
- The output from Affine Network is used in Spatial Transform Affine to generate a cropped image with higher activation
- The cropped image is fed into TResNet-M with pre-trained ImageNet weights which is used as feature extractor
- Feature extractor is fully connected to Global-K-Max-Pooling followed by a Dense, Batch Normalization and Dense layer. 
- The output of the above is fed through a L2 Normalized Dense layer for classification 
## Loss Function
- [Li-ArcFace Loss](https://arxiv.org/abs/1907.12256)
  - Stage 1, 40.0 scale, no margin
  - Stage 2, 40.0 scale, 0.15 margin
  - Stage 3, 30.0 scale, 0.3 margin
## Metrics
- Sparse Categorical Accuracy
## Optimizer
- [AdamW](https://arxiv.org/pdf/1711.05101.pdf)
## Train Loop Configuration
- Attention Net and Classifier weights are ImageNet pre-trained 
- Multi Stages
  - Stage 1, 30 epochs
  - Stage 2, 30 epochs
  - Stage 3, 30 epochs
### Callbacks
#### Learning Rate Scheduler
- [Cosine Decay with Restarts](https://arxiv.org/pdf/1608.03983.pdf)
#### KTF Variable Scheduler
- ktf.train.callbacks.VarScheduler
  - Used to schedule the AdamW `weight_decay` params. Implementation of [AdamW in tfa](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW) assumes constant learning rate, so we have to do it manually.
#### Model Checkpoint
- Save best validation model
#### CSV Logger
- To log training progress