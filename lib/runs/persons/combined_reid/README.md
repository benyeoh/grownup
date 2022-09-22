# Config Files of Experimental Runs of Combined Reid

## Multi Source
Multi Source is a dataset consisting of the following Person ReID datasets:
* [CUHK-SYSU (also known as PersonSearch)](http://www.ee.cuhk.edu.hk/~xgwang/PS/paper.pdf)
* [CUHK02](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Li_Locally_Aligned_Feature_2013_CVPR_paper.pdf)
* [CUHK03](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf)
* [DukeMTMC ReID](https://github.com/sxzrt/DukeMTMC-reID_evaluation)
* [Market 1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)

The breakdown of ids and images per subset is as follows:  
| subset        | # ids | # images |
|---------------|-------|----------|
| cuhk_sysu     | 11934 | 34574    |
| cuhk02        | 1816  | 7264     |
| cuhk03        | 1467  | 14097    |
| dukemtmc_reid | 1812  | 36411    |
| market_1501   | 1501  | 29419    |
| **total**     |**18530**|**121765**|

The expected baseline accuracy for Multi Source using DualNorm with ResNet-50 backbone and image size 256 x 128 is 81.1%

Config File | Experiment Accuracy | Expected Accuracy
----------- | ------------------- | -----------------
[dual_norm_baseline](multi_source/dual_norm_baseline/config.json) | 80.4% | 81.1%
