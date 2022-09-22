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
- As per the [paper](https://arxiv.org/pdf/2005.05123v1.pdf), Attention Network is made out of the first few blocks of ResNet50 while Affine Network is made out of 2 small multi-layer-perceptrons (MLP)
- The output from Affine Network is used in Spatial Transform Affine to generate a cropped image with higher activation
- The cropped image is fed into TResNet-M with pre-trained ImageNet weights which is used as feature extractor
- Feature extractor is fully connected to an Embedding component which outputs a feature vector for classification and an L2-normalized feature vector for Embedding Loss
- The feature vector from the Embedding component is connected to a Dense layer for classification
## Loss Function
- Sparse Categorical Cross Entropy
  - In the interest of memory usage reduction, dataset labels are integers and a sparse variant of Categorical Cross Entropy loss is used
- Embedding Loss
  - This loss is used to encourage closer distances between feature vectors of the same class and farther distances for others.
## Loss Weights
- The ratio of value of Embedding Loss to CE Loss is 2:1 as per the [paper](https://arxiv.org/pdf/2005.05123v1.pdf)
## Metrics
- Sparse Categorical Accuracy
## Optimizer
- [AdamW](https://arxiv.org/pdf/1711.05101.pdf)
## Train Loop Configuration
- Epochs = 310
- Attention Net and Classifier weights are ImageNet pre-trained 
### Callbacks
#### Learning Rate Scheduler
- [Cosine Decay with Restarts](https://arxiv.org/pdf/1608.03983.pdf)
  - Initial LR = 0.0005
  - First Decay Steps = 10
  - Decay Steps multiplier = 2
  - Learning Rate reset multiplier = 0.9
#### KTF Variable Scheduler
- ktf.train.callbacks.VarScheduler
  - Used to schedule the AdamW `weight_decay` params. Implementation of [AdamW in tfa](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW) assumes constant learning rate, so we have to do it manually.
  - Cosine Decay with Restarts schedule
    - Initial LR = 0.0003
    - First Decay Steps = 10
    - Decay Steps multiplier = 2
    - Weight Decay reset multiplier = 0.9
#### Model Checkpoint
- Save best validation model
#### CSV Logger
- To log training progress