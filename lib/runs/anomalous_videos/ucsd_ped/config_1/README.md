# Config File Description
## Datasets
### Sources
- Train Dataset is taken from `hpc-datasets/anomalous_videos/ucsd/ucsd_ped_2/tfrecord/Train`
  - Shuffled and repeated
  - Batch size is set to 18 to maximize single GPU memory on DGX
- Validation Dataset is taken from `hpc-datasets/anomalous_videos/ucsd/ucsd_ped_2/tfrecord/Val`
  - A video from original train dataset with the least number of frames
  - Dataset is repeated
  - Batch size is set to 18 to maximize single GPU memory on DGX
### Augmentations
- Both datasets are standardized to 0.5 mean and 0.5 standard deviation
## Model
- A sequence of 16 frames enters the encoder
- The encoder produces an encoded representation of the input. 
  - Note that Spatial Dropout 3D is applied where appropriate.
- A "cosine similarity" measure is calculated between the encoded representation and every memory slot
- A new encoded representation is formed via weighted average of the memory slots where the weights are generated from the "cosine similarity" measures
- The new encoded representation enters the decoder to produce a reconstruction of the input 16 frames
  - Note that Spatial Dropout 3D is applied where appropriate.
## Loss Functions
- There are 2 loss functions:
  - Reconstruction L2 Loss
    - Measures L2 difference for each pixel between input sequence of frames and reconstructed output of frames
  - Memory Entropy Loss 
    - Applies entropy loss to the memory weights to encourage sparsity of weights. This minimises amt of memory slots used leading to higher reconstruction errors for abnormal frames. 
  - Weightage of Memory Entropy Loss to Reconstruction L2 Loss is 6e-04 to 1
## Metrics
- None
## Optimizer
- [Ranger](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/RectifiedAdam)
  - LR: 1e-04
  - epsilon: 1e-08
  - Default settings for the rest
## Train Loop Configuration
- 160 epochs
- Number of training steps and validation steps are set to ensure the whole dataset is trained on in 1 epoch
### Callbacks
#### Model Checkpoint
- Save best validation model
#### CSV Logger
- To log training progress
