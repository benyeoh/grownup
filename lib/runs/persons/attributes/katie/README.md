# Config Files of Experimental Runs of DeepMAR on Katie Persons Dataset(s)

## KATIE
* Datasets
     * [List](https://jira.internal.klass.dev/browse/QUEEN-165)
     * Annotated and exported from [KATIE](https://katietrial.klassengineering.com.sg/home) 
* [Annotation Guide](https://confluence.internal.klass.dev/display/QUE/Label+and+attributes+-+Person)

`katie/config.json` serves as a base config to do transfer learning on multiple `KATIE Persons` dataset via `concatenate`.

Training configurations are adapted from `peta/config.json`.

The following will need to be configured accordingly for training with different datasets.
- `num_steps` 
- `num_valid_steps` 
- `class_weights` 
- `num_outputs` 
- `metrics`'s `num_classes`

| Config File                | Experiment Accuracy |
| -------------------------- | ------------------- |
| [katie](katie/config.json) | 76.2 (Resnet50v2)   |

This is just a benchmark for the datasets used in config provided.
With subset of 7 labels, using as `/hpc-datasets/person_retrieval/dataset/katie/labels/labels_subset_carrying.json` as `tf_mapping_labels.json`

