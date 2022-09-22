# Config Files of Experimental Runs of TUT-SED Synthetic 2016

[TUT-SED Synthetic 2016](https://webpages.tuni.fi/arg/paper/taslp2017-crnn-sed/tut-sed-synthetic-2016) is a dataset consisting 100 synthetically generated audio clips, containing 10 classes of sound events.

SOTA and our training results can be found here:
https://confluence.internal.klass.dev/display/QUE/Sound+Event+Detection+Training+Summary


Config File | Validation F1 (micro) | Test F1 (micro) | Expected Test F1 (micro) |
----------- | ------------------- | ------------- | ---------------------- |
[DepthWise Separable and Dilated Convolutions (dnd_baseline)](baseline/config.json) | 69.48% | 64.69% | 63 +- 2%
[dnd_baseline + conformer](conformer_baseline/config.json) | 70.67% | 69.29% | NIL |
[dnd_baseline + conformer + mean_teacher](mean_teacher/config.hjson) | 71.32% | 71.38% | NIL |
