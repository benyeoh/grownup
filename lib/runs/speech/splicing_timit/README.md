# Config Files and Experimental Results on the Spliced TIMIT database

Config File: [audio_splicing_seres2net34.json](audio_splicing_seres2net34.json)

Data Type | Eval Set Sparse Categorical Acc. | Description
----------| ---------------------------------| -------------
Clean data | 83.79 | Spliced database was generated using original TIMIT data
Aircon 25dB | 93.62 | Added aircon noise into TIMIT data at SNR level 25dB then generated spliced data
Aircon 15dB | 94.36 | Added aircon noise into TIMIT data at SNR level 15dB then generated spliced data

# Research work on SE-Res2Net-Conformer
The Config File is [audio_splicing_seres2net34_conformer.hjson](audio_splicing_seres2net34_conformer.hjson). The preliminary results on test sets were:

Test Set | Sparse Categorical Accuracy
---------|-----------------------------
Clean data | 87.14
Aircon 15dB | 96.10
Aircon 25dB | 95.72
Filed noise | 92.37

A few parameters can be tuned, e.g. `scale` in `ktf.models.networks.seres2net34` can be either 4 or 8; `head_size` and `num_heads` in `ConformerBlock` can be reduced.

The tfrecord files can be found at: `/hpc-datasets/wanglei/data/timit`.
