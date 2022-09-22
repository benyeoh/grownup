# Config Files and Experimental Results of ASVspoof2019 Countermeasures

Config File | Eval Set Accuracy | EER   | Description
----------- | ----------------- | ------| ----------------
[tresnet34_onehead.json](tresnet34_onehead.json) | ~ 17% | ~ 8.28% | Baseline with 2 classes (`genuine` vs `spoofing`)
[seres2net34_onehead.json](seres2net34_onehead.json) | - | ~ 5% | SE-Res2Net with 2 classes 
[seres2net34_multiclasses.json](seres2net34_multiclasses.json) | - | ~ 2.12% | SE-Res2Net with 3 classes but EER is based on 2 classes (by merging `TTS` and `VC`)*

Note: 
(1) For 3 classes, they are `genuine`, `TTS` and `VC`. In another words, `spoofing` is the combination of `TTS` and `VC`.

(2) Equal error rate (EER) is a biometric security system algorithm used to predetermine the threshold values for its false acceptance rate and its false rejection rate. When the two rates are equal, the common value is referred to as the equal error rate. The value indicates that the proportion of false acceptances is equal to the proportion of false rejections. The lower the equal error rate value, the higher the accuracy of the biometric system. (https://www.webopedia.com/definitions/equal-error-rate/)

The EER scoring tool used in this experiment is the official script provided by ASVspoof committee (https://www.asvspoof.org/)


# Research work on SE-Res2Net-Conformer
The Config File is [seres2net34_conformer.hjson](seres2net34_conformer.hjson). The preliminary result on Eval set was about 94.6% as Sparse Categorical Accuracy. A few parameters can be tuned, e.g. `scale` in `ktf.models.networks.seres2net34` can be either 4 or 8; `head_size` and `num_heads` in `ConformerBlock` can be reduced.

The tfrecord files can be found at: `/hpc-datasets/wanglei/data/asvspoof2019/tfrecord_multi`.
