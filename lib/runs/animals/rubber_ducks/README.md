# Config Files for runs on the Rubber Ducks Dataset 
The Rubber Ducks dataset is a toy dataset provided with the Tensorflow Object Detection API (TF OD API). It contains a single rubber ducks class for the purpose of fine tuning and debugging object detection models.

At the moment, the KTF integration of the TF OD API does not support training metrics. In place of a metric, we use the final training loss of the model as a gauge of how well the model has been trained. 

Note that this loss is the sum of the `localization_loss` and `classification_loss`.

The dataset can be found here: `klassterfork/python/third_party/tf_model_garden/research/object_detection/test_images/ducky`


Filename | Config | Final Training Loss | Expected Final Training Loss |
-------- | ------ | ------------------- | ----------------------------- |
`reference.hjson` |[ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 with hand-modified pretrained weights](reference.hjson) | ~0.00336 | ~0.00323 |
`simple.hjson` | [ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 with hand-modified pretrained weights](simple.hjson) | 0.67680 | N/A |

* For the `reference.hjson` config, we use the same exact set of hyperparameters in the [tutorial of the TF OD API](https://colab.research.google.com/github/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb) and check that we are able to train to the same final loss.

* On the other hand, `simple.hjson` is just the simplest train config that can converge to an okay-ish result, mainly used to demonstrate how to use the TF OD API integration in KTF.
