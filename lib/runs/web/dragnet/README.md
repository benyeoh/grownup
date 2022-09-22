# Table 1: Config Files of Experimental Runs of Dragnet Dataset

| Config File                                                                               | Experiment Accuracy (%) | Description |
| ----------------------------------------------------------------------------------------- | ----------------------- | ----------- |
| [conv_graph_net/config](conv_graph_net/config.json)                                       | ~97                     | -           |
| [conv_graph_net/config_10L_json_graph](conv_graph_net/config_10L_json_graph.json)         | ~97.61                  | -           |
| [conv_graph_net/config_30k_cc_2008_decode](conv_graph_net/config_30k_cc_2008_decode.json) | ~97.6                   | -           |

# Table 2: Training Config of Dragnet Dataset with Different Pre-train Weights Applied (Best F1 on testset presented)

| Config File                                                                                          | GROWN+UP 1.0 | GROWN+UP 2.0 w/ visual feat. | GROWN+UP 2.0 w/ xform | Description                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------- | ------------ | ---------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [conv_graph_net/config_10L_no_pt.json](conv_graph_net/config_10L_no_pt.json)                         | ~ 0.946      | ~ 0.938                      | -                     | Conv graph model with default architecture (S=10, K=256). No pre-train applied                                                                                                                                                                |
| [conv_graph_net/config_10L_mask_only_cc_2008.json](conv_graph_net/config_10L_mask_only_cc_2008.json) | ~ 0.951      | -                            | -                     | Conv graph model with default architecture (S=10, K=256). Pre-trained from CommonCrawl 2008 45k subset with 1 objective: masked nodes prediction                                                                                              |
| [conv_graph_net/config_10L_full_pt_cc_2008.json](conv_graph_net/config_10L_full_pt_cc_2008.json)     | ~ 0.952      | ~ 0.950                      | -                     | Conv graph model with default architecture (S=10, K=256). Pre-train from Commoncrawl 2008 45k (for grownup 1.0) and Commoncrawl 2008 88k (for grownup 2.0 w/ visual feat.) subsets with 2 objectives: masked node and same website prediction |

