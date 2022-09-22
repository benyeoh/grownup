# Config Files of Experimental Runs on AI City

AI City is a dataset containing 52717 images, 440 vehicle IDs, 40 camera views, and 2173 tracks.

The dataset used for training is a subset of the above dataset, containing 34605 images and 352 vehicle IDs.
A 20% random split on this subset is used as the validation set.

This is not an exact match to the [paper's](https://github.com/michuanhaohao/AICITY2021_Track2_DMT/blob/50f27363532ae712868ff1ceaf128a3bbec426ac/paper.pdf), but it is the closest we could get to it.

The dataset split can be found at: /hpc-datasets/cars/ai_city/raw/full_sized/labels/split_by_track

Evaluation of the model's performance is done in Vehid POC, a vehicle reID model.

Config File | Experiment Validation Accuracy |
----------- |------------------------------------- |
[AI City](config.json) | 98.52% | 
