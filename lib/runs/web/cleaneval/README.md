# Table 1: Config Files of Experimental Runs of CleanEval Dataset

| Config File                                                                             | Experiment Accuracy (%) | Description                                   |
| --------------------------------------------------------------------------------------- | ----------------------- | --------------------------------------------- |
| [conv_graph_net/config.json](conv_graph_net/config.json)                                | ~ 82                    | Baseline                                      |
| [conv_graph_net/config_10L_dragnet.json](conv_graph_net/config_10L_dragnet.json)        | ~ 83.7                  | Pre-trained from dragnet dataset + 10 layers  |
| [conv_graph_net/config_15k_cc_2008.json](conv_graph_net/config_15k_cc_2008.json)        | ~ 84.5                  | Pre-trained from CommonCrawl 2008 ~15k subset |
| [conv_graph_net/config_30k_cc_2008.json](conv_graph_net/config_30k_cc_2008_decode.json) | ~ 85                    | Pre-trained from CommonCrawl 2008 ~30k subset |


# Table 2: Training Config of CleanEval Dataset with Different Pre-train Weights Applied (Best F1 on testset presented)

| Config File                                                                                          | GROWN+UP 1.0 | GROWN+UP 2.0 w/ visual features | GROWN+UP 2.0 w/ xform | Description                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------- | ------------ | ------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [conv_graph_net/config_10L_no_pt.json](conv_graph_net/config_10L_no_pt.json)                         | ~ 0.834      | ~ 0.837                         | -                     | Conv graph model with default architecture (S = 10, K = 256). No pre-train applied                                                                                                                                                            |
| [conv_graph_net/config_10L_mask_only_cc_2008.json](conv_graph_net/config_10L_mask_only_cc_2008.json) | -            | -                               | -                     | Conv graph model with default architecture (S = 10, K = 256). Pre-trained from CommonCrawl 2008 45k subset with 1 objective: masked nodes prediction                                                                                          |
| [conv_graph_net/config_10L_ful_pt_cc_2008.json](conv_graph_net/config_10L_full_pt_cc_2008.json)      | ~ 0.865      | ~ 0.860                         | -                     | Conv graph model with default architecture (S=10, K=256). Pre-train from Commoncrawl 2008 45k (for grownup 1.0) and Commoncrawl 2008 88k (for grownup 2.0 w/ visual feat.) subsets with 2 objectives: masked node and same website prediction |
