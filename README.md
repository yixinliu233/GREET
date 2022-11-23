# GREET

This is the source code of AAAI'23 paper "Beyond Smoothing: Unsupervised Graph Representation Learning with Edge Heterophily Discriminating".

## Requirements
This code requires the following:
* Python==3.9
* Pytorch==1.11.0
* Pytorch Geometric==2.0.4
* DGL==??
* Numpy==1.21.2
* Scikit-learn==1.0.2
* Scipy==??

## Usage
Just run the script corresponding to the experiment and dataset you want. For instance:

* Run on Cora dataset:
```
bash script/run_cora.sh
```

## Cite

If you compare with, build on, or use aspects of this work, please cite the following:
```
@inproceedings{liu2023goodd,
  title={GOOD-D: On Unsupervised Graph Out-Of-Distribution Detection},
  author={Liu, Yixin and Ding, Kaize and Liu, Huan and Pan, Shirui},
  booktitle={Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
  year={2023}
}
```
