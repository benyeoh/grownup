# Training Config of Stabilo Dataset (Best F1 on testset presented)

| Config File                                                                              | w/ visual features + w/ xform | Description                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [conv_graph_net/config_trans_vfeats_pt.json](conv_graph_net/config_trans_vfeats_pt.json) | 0.948                         | Conv graph model with transformer block (S=5, T=5). Full pre-train (from Commoncrawl 2022 50k webpages) applied. Topic: forum (1400+ webpages). 50% for train/valid and 50% for test |
