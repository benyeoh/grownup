# Config Files of Experimental Runs of Mivia Audio Events

[Mivia Audio Events](https://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/#:~:text=The%20dataset,(composed%20of%201800%20events).) composed of a total of 6000 events for surveillance applications, namely glass breaking, gun shots and screams. 

The 6000 events are divided into a training set (composed of 4200 events) and a test set (composed of 1800 events). 

The data set is designed to provide each audio event at 6 different values of signal-to-noise ratio (namely 5dB, 10dB, 15dB, 20dB, 25dB and 30dB) and overimposed to different combinations of environmental sounds in order to simulate their occurrence in different ambiences.

SOTA and our training results can be found here:
https://confluence.internal.klass.dev/display/QUE/Sound+Event+Detection+Training+Summary

Config File | Validation Top-1 Recall | Expected Top-1 Recall |
----------- | ------------------- | ------------- |
[dnd_baseline + conformer](conformer_baseline/config.json) | 99.41 | NIL |
[dnd_baseline + conformer + mean_teacher 20% Labelled 80% Unlabelled](mean_teacher/config.hjson) | 99.48 | NIL |
