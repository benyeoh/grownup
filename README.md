# GROWN+UP: A "Graph Representation Of a Webpage" Network Utilizing Pre-Training
This is the official repo of GROWN+UP and accompanying benchmarks published in the [CIKM'22 proceedings](https://dl.acm.org/doi/10.1145/3511808.3557340). Latest preprint can be found on [arxiv](https://arxiv.org/abs/2208.02252).

## Pre-requisites
The hardware / software requirements are:

* Ubuntu 18.04 or newer
* Docker 19.03 or newer with GPU support
  * (Optional but recommended)
* NVIDIA GPU with CUDA 11.2.1 support (GPU driver version: >=460.32.03)
  * Typically, you don't need to care on Cuda library installation if you use Docker
* [Git LFS](https://git-lfs.github.com/)
  * This is **super important and is required** when you clone / pull from this repo, since some large data (ie, pretrained weights) is stored in LFS.

## Introduction
The top level folder structure of this repo consists of:

1. `boilerplate-removal`
Webpage boilerplate removal benchmarks for GROWN+UP as well as other baselines mentioned in the paper.
2. `genre-classification`
Webpage genre classification benchmarks for GROWN+UP
3. `klassterfork`
A subset of an ML framework containing GROWN+UP model components and other ML training necessities to reproduce results, built on Tensorflow v2.5
4. `pre-training` TODO

For more details, please consult the README.md in the appropriate folders.

## Citation
To cite, please use this BibTex:

```
@inproceedings{10.1145/3511808.3557340,
author = {Yeoh, Benedict and Wang, Huijuan},
title = {GROWN+UP: A ''Graph Representation Of a Webpage" Network Utilizing Pre-Training},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557340},
doi = {10.1145/3511808.3557340},
booktitle = {Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
pages = {2372–2382},
numpages = {11},
keywords = {web genre classification, webpage, boilerplate removal, feature extractor, self-supervised, graph neural network, backbone, pre-training},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}
```




