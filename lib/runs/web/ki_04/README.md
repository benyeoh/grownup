# Config Files of Experimental Runs of KI-04 Dataset

| Config File                                                                                          | F1 on Testset   | Description                                                                                                                                                         |
| ---------------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [conv_graph_net/config_7L_no_pt.json](conv_graph_net/config_7L_no_pt.json)                           | -               | Conv graph model with less layer (S = 5, K = 64). No pre-train applied                                                                                              |
| [conv_graph_net/config_7L_full_pt_cc_2008.json](conv_graph_net/config_7L_full_pt_cc_2008.json)       | -               | Conv graph model with less layer (S = 5, K = 64). Pre-trained from CommonCrawl 2008 45k subset with 2 objectives: masked node and same website prediction           |
| [conv_graph_net/config_10L_mask_only_cc_2008.json](conv_graph_net/config_10L_mask_only_cc_2008.json) | ~ 0.73 - 0.85 * | Conv graph model with default architecture (S = 10, K = 256). Pre-trained from CommonCrawl 2008 45k subset with 1 objective: masked nodes prediction                |
| [conv_graph_net/config_10L_full_pt_cc_2008.json](conv_graph_net/config_10L_full_pt_cc_2008.json)     | ~ 0.74 - 0.84 * | Conv graph model with default architecture (S = 10, K = 256). Pre-train from Commoncrawl 2008 45k subset with 2 objectives: masked node and same website prediction |

**<*>** There is a variance of the F1 score depending on the split of the dataset
