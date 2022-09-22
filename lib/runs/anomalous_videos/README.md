# Config Files of Experimental Runs of Video Anomaly Detection Datasets
## Note:
- The training config experimented on is slightly different from the [official](https://github.com/donggong1/memae-anomaly-detection) and [unofficial](https://github.com/lyn1874/memAE) implementations
## Datasets
### [UCSD_Ped_2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)
- Note: Original implementation's provided weights gave AUC of ~91% although paper claims 94.1%

Config File | Experiment AUC | [Official Implementation](https://github.com/donggong1/memae-anomaly-detection) | [Unofficial Implementation](https://github.com/lyn1874/memAE)
----------- | -------------- | ----------------------- | ------------------------- 
[UCSD_Ped_2_Config_1](ucsd_ped/config_1/config.json)| 92.01% | 94.1% | 94.0%

### [Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
Config File | Experiment AUC | [Official Implementation](https://github.com/donggong1/memae-anomaly-detection) | [Unofficial Implementation](https://github.com/lyn1874/memAE)
----------- | -------------- | ----------------------- | ------------------------- 
[Avenue_Config_1](avenue/config_1/config.json)| 81.92% | 81% | 83.3%

### [Shanghai Tech](https://svip-lab.github.io/dataset/campus_dataset.html) 
- The original paper states an AUC of 71.2%. However, I was unable to replicate the AUC when I trained the full dataset. Therefore, I decided to train for each background separately. Below are the results:

Config File | Experiment AUC | [Official Implementation](https://github.com/donggong1/memae-anomaly-detection) | [Unofficial Implementation](https://github.com/lyn1874/memAE)
----------- | -------------- | ----------------------- | ------------------------- 
[Shanghai_Tech_Config_01](shanghai_tech/config_01/config.json)| 86.36% | - | -
[Shanghai_Tech_Config_02](shanghai_tech/config_02/config.json)| 71.62% | - | -
[Shanghai_Tech_Config_03](shanghai_tech/config_03/config.json)| 76.49% | - | -
[Shanghai_Tech_Config_04](shanghai_tech/config_04/config.json)| 56.15% | - | -
[Shanghai_Tech_Config_05](shanghai_tech/config_05/config.json)| 72.03% | - | -
[Shanghai_Tech_Config_06](shanghai_tech/config_06/config.json)| 87.80% | - | -
[Shanghai_Tech_Config_07](shanghai_tech/config_07/config.json)| 59.81% | - | -
[Shanghai_Tech_Config_08](shanghai_tech/config_08/config.json)| 77.01% | - | -
[Shanghai_Tech_Config_09](shanghai_tech/config_09/config.json)| 87.23% | - | -
[Shanghai_Tech_Config_10](shanghai_tech/config_10/config.json)| 49.54% | - | -
[Shanghai_Tech_Config_11](shanghai_tech/config_11/config.json)| 92.23% | - | -
[Shanghai_Tech_Config_12](shanghai_tech/config_12/config.json)| 63.24% | - | -
- All in all, the above experiments lead to a weighted average (weighted by number of test videos per background) AUC of 74.64%
