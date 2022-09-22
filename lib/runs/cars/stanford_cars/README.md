# Config Files of Experimental Runs of Stanford Cars

Stanford Cars is a dataset of 196 car models with 8000+ images in train and test set respectively

The expected baseline accuracy for Stanford Cars using ResNet50V2 with image size 448 x 448 is 92.7%

Config File | Experiment Accuracy | Expected Accuracy
----------- | ------------------- | -----------------
[ResNet50v2_baseline_resize_aspect_ratio_preserved](baseline_resize_aspect_ratio_preserved/config.json) | 91.742% | 92.7%
[ResNet50v2_baseline](baseline/config.json) | 91.842% | 92.7%
[ResNet50v2_baseline_improved](baseline_improved/config.json) | 93.01% | 92.7%
[ResNet50v2_baseline_arcloss](baseline_arcloss/config.json) | 93.8% | 92.7%
[TResNetM_baseline](tresnetm_baseline/config.json) | 93.570% | no benchmark
[Att-Aff](att-aff/config.json) | 94.98% | no benchmark
[Att-Aff-6params](att-aff_improved/config.json) | 95.25% | no benchmark

Note:
- Benchmark is not provided for Att-Aff as the classifier network used is ResNet-101 which is different from our choice of network (TResNet-M)