# Config File Description
## Datasets
### Sources
- Train Dataset is taken from `hpc-datasets/cars/stanford_car/original_kaggle/tfrecord_600_resize_with_pad/train`
  - Sized 600 x 600 from original image with aspect ratio preservation
  - Shuffled
  - Training batch size is set to 16 as per standard configuration in literature
- Validation Dataset is taken from `hpc-datasets/cars/stanford_car/original_kaggle/tfrecord_600_resize_with_pad/test`
  - Sized 600 x 600 from original image with aspect ratio preservation
  - Validation batch size is set to 512 to maximize GPU usage and quicken validation accuracy processing
### Augmentations
- Train Dataset is 
  - Various augmentations using `albumentations`
  - Central cropped to size 448 
  - Normalized to ImageNet mean/var
- Validation Dataset is 
  - Central cropped to size 448
  - Normalized to ImageNet mean/var
## Model
- ResNet50V2 with pre-trained ImageNet weights is used as feature extractor
- Feature extractor is fully connected to Dense layer as a head for classification
## Loss Function
- [Li-ArcFace Loss](https://arxiv.org/abs/1907.12256)
  - Stage 1, 40.0 scale, no margin
  - Stage 2, 40.0 scale, 0.25 margin
  - Stage 3, 30.0 scale, 0.4 margin
## Metrics
- Sparse Categorical Accuracy
## Optimizer
- [AdamW](https://arxiv.org/pdf/1711.05101.pdf)
## Train Loop Configuration
- Multi Stages
  - Stage 1, 30 epochs
  - Stage 2, 30 epochs
  - Stage 3, 30 epochs
### Callbacks
#### Learning Rate Scheduler
- [Cosine Decay with Restarts](https://arxiv.org/pdf/1608.03983.pdf)
  - Stage 1, 0.00025 lr
  - Stage 2, 0.000125 lr
  - Stage 3, 0.00008 lr
#### Model Checkpoint
- Save best validation model