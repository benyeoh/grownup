# Config File Description
## Datasets
### Sources
- Train Dataset is taken from `hpc-datasets/multi_objects/cifar10/tfrecords/train`
  - Sized 224x224 image with batchsize 16.
- Validation Dataset is taken from `hpc-datasets/multi_objects/cifar10/tfrecords/test`
  - Sized 224x224 image with batchsize 16.
### Augmentations
- Train Dataset is 
  - Random flipped left right 
  - Random color jitter.
  - Standardized according to ImageNet color mean and standard deviation
- Validation Dataset is 
  - Random flipped left right.
  - Random color jitter.
  - Standardized according to ImageNet color mean and standard deviation
## Model
- Encoder:
  - Resnet50 with ImageNet pretrained weights. 
  - Followed by a 2 layer MLP as embedding layer.
- Predictor:
  - 2 layer MLP to predict the embedding of differently augmented image.
## Loss Function
- GradStopCosineSimilarity
  - Cross branch cosine similarity with gradient stop applied on output by encoder
## Optimizer
- SGD with momentum
  - LR = 0.01
  - Momentum = 0.6
## Train Loop Configuration
- Epochs = 200
### Callbacks
#### Learning Rate Scheduler
- Cosine Decay
  - Initial LR = 0.01
  - Decay steps = 500
#### Model Checkpoint
- Save best validation model