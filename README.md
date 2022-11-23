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

* Run out-of-distribution detection on BZR (ID) and COX2 (OOD) datasets:
```
bash script/oodd_BZR+COX2.sh
```

* Run anomaly detection on PROTEINS_full datasets:
```
bash script/ad_PROTEINS_full.sh
```
