# Config Files of Experimental Runs of DeepMAR

## PETA
* [Dataset](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html#:~:text=The%20PETA%20dataset%20consists%20of,and%204%20multi%2Dclass%20attributes.)
     * Note : the one used is DeepMAR format, not the original dataset
* [Paper](http://mmlab.ie.cuhk.edu.hk/projects/PETA_files/Pedestrian%20Attribute%20Recognition%20At%20Far%20Distance.pdf)


| Config File              | Experiment Accuracy | Expected Accuracy                   |
| ------------------------ | ------------------- | ----------------------------------- |
| [peta](peta/config.json) | 83.6% (Resnet50)    | 82.6% (CaffeNet) - 84.36%(Resnet50) |


#### Statistics

| Partition            | TRAINVAL | Weight | TEST | Weight |
| -------------------- | -------- | ------ | ---- | ------ |
| Total # Images       | 11400    | -      | 7600 | -      |
| Total # Attributes   | 35       | -      | 35   | -      |
| personalLess30       | 5612     | 0.49   | 3831 | 0.5    |
| personalLess45       | 3806     | 0.33   | 2441 | 0.32   |
| personalLess60       | 1165     | 0.1    | 779  | 0.1    |
| personalLarger60     | 711      | 0.06   | 461  | 0.06   |
| carryingBackpack     | 2238     | 0.2    | 1497 | 0.2    |
| carryingOther        | 2258     | 0.2    | 1531 | 0.2    |
| lowerBodyCasual      | 9810     | 0.86   | 6546 | 0.86   |
| upperBodyCasual      | 9712     | 0.85   | 6495 | 0.85   |
| lowerBodyFormal      | 1569     | 0.14   | 1043 | 0.14   |
| upperBodyFormal      | 1538     | 0.13   | 1007 | 0.13   |
| accessoryHat         | 1205     | 0.11   | 725  | 0.1    |
| upperBodyJacket      | 780      | 0.07   | 534  | 0.07   |
| lowerBodyJeans       | 3500     | 0.31   | 2315 | 0.3    |
| footwearLeatherShoes | 3413     | 0.3    | 2215 | 0.29   |
| upperBodyLogo        | 467      | 0.04   | 296  | 0.04   |
| hairLong             | 2709     | 0.24   | 1804 | 0.24   |
| personalMale         | 6274     | 0.55   | 4147 | 0.55   |
| carryingMessengerBag | 3344     | 0.29   | 2275 | 0.3    |
| accessoryMuffler     | 962      | 0.08   | 632  | 0.08   |
| accessoryNothing     | 8496     | 0.75   | 5742 | 0.76   |
| carryingNothing      | 3152     | 0.28   | 2091 | 0.28   |
| upperBodyPlaid       | 302      | 0.03   | 203  | 0.03   |
| carryingPlasticBags  | 868      | 0.08   | 585  | 0.08   |
| footwearSandals      | 244      | 0.02   | 144  | 0.02   |
| footwearShoes        | 4128     | 0.36   | 2774 | 0.36   |
| lowerBodyShorts      | 399      | 0.04   | 261  | 0.03   |
| upperBodyShortSleeve | 1605     | 0.14   | 1090 | 0.14   |
| lowerBodyShortSkirt  | 513      | 0.04   | 351  | 0.05   |
| footwearSneaker      | 2431     | 0.21   | 1675 | 0.22   |
| upperBodyThinStripes | 200      | 0.02   | 126  | 0.02   |
| accessorySunglasses  | 334      | 0.03   | 218  | 0.03   |
| lowerBodyTrousers    | 5838     | 0.51   | 3949 | 0.52   |
| upperBodyTshirt      | 960      | 0.08   | 640  | 0.08   |
| upperBodyOther       | 5200     | 0.46   | 3456 | 0.45   |
| upperBodyVNeck       | 144      | 0.01   | 80   | 0.01   |
