# Config Files of Experimental Runs on Combined Cars

Combined Cars is an ever-growing dataset with undefined number of labels. It comprises various datasets, merged to form a larger, more comprehensive dataset.

There is no expected baseline accuracy for this dataset, but a benchmark is available to compare across different combined datasets.

Benchmarked on Stanford, CompCars, VMMRDb & Traffic Images merged dataset holdout set:

Holdout set path: `/hpc-datasets/cars/combined_cars/tfrecords/labels_scvt/stratified_split_448_resize_without_pad/test`

Config File | Validation Set | Experiment Validation Accuracy | Test Accuracy on holdout set for SCVT |
----------- | -------------- | ------------------------------ | ------------------------------------- |
[Att-Aff on Stanford, CompCars & VMMRDb (SCV)](att-aff/SCV/config.json) | Validation set on SCV | 93.9% | 92.2% | 
[Att-Aff on Stanford, CompCars, VMMRDb & Traffic Images (SCVT)](att-aff/SCVT/config.json) | Validaton set on SCVT | 94.3% | 93.2% |
