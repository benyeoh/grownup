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
  - Random cropped to size 448
  - Random flipped left right 
  - Standardized according to ImageNet color mean and standard deviation
- Validation Dataset is 
  - Central cropped to size 448
  - Standardized according to ImageNet color mean and standard deviation
## Model
- ResNet50V2 with pre-trained ImageNet weights is used as feature extractor
- Feature extractor is fully connected to Dense layer as a head for classification
- Dense layer for classification is regularized with L2 = 0.0005
## Loss Function
- Sparse Categorical Cross Entropy
  - In the interest of memory usage reduction, dataset labels are integers and a sparse variant of Categorical Cross Entropy loss is used
## Metrics
- Sparse Categorical Accuracy
## Optimizer
- SGD with momentum
  - LR = 0.005
  - Momentum = 0.9
## Train Loop Configuration
- Epochs = 300
### Callbacks
#### Learning Rate Scheduler
- [Cosine Decay with Restarts](https://arxiv.org/pdf/1608.03983.pdf)
  - Initial LR = 0.005
  - First Decay Steps = 10
  - Decay Steps multiplier = 2
#### Model Checkpoint
- Save best validation model