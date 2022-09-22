# Config Files for runs on the Oxford IIIT Pets dataset
Oxford-IIIT Pets is a dataset that consists of 37 categories of cats and dogs with roughly 200 images for each pet breed. 

The dataset home page is https://www.robots.ox.ac.uk/~vgg/data/pets/

## Object Detection
Filename | Config | Final Training Loss |
-------- | ------ | ------------------- |
`object_detection.hjson` | [ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 with pretrained weights](object_detection.hjson) | 0.90317 |

## Image Segmentation
Filename | Config | Final Training Loss |
-------- | ------ | ------------------- |
`instance_segmentation.hjson` | [tf2_mask_rcnn_inception_resnet_v2_pets ](image_segmentation.hjson) | 0.21223 |
