# SNSM
Source Code for Self-Neighbourhood Support Maps Project
This repository provides the implementation of the [SNSM](https://ieeexplore.ieee.org/abstract/document/10889993) approach. Some of the code snippets are stolen from [AnyLoc](https://github.com/AnyLoc/AnyLoc.git) and [Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD.git) works.  

## Summary
This work proposes a new training-free technique for the all-day Visual Place Recognition (VPR) problem. It especially addresses the illumination challenge in the VPR. The SNSM function accepts a feature block from the backbone model and aggregates it into a modality-invariant (RGB and thermal) feature map called an SNSM map. Essentially, this attempts to capture the support value of each selected patch from its neighbourhood, which retains the homogeneous structural details and suppresses the heterogeneous modality-specific features. For further details, please refer to the full paper. The simple and training-free SNSM improve upon popular VPR models and various unsupervised methods by a considerable margin.       

## Repo. details
The SNSM directory contains the All-Backbones-VLAD_bl.py file, which can produce recall rates for the choice of dataset and aggregator. Currently, only unsupervised feature extraction techniques are included. However, VPR models reported in the paper are off-the-shelf, and models are open-source. 
## Data
Inference datasets: [RGB-T Datasets Drive link](https://drive.google.com/file/d/11qBQw9DadQ5MemUTouK6HolPEPCo-BXT/view?usp=sharing) 

## Cluster
Make appropriate changes to the directory paths in the code snippets. 
The below command generates clusters to initialize the model. 
```python
python main.py --mode=cluster
```
## Train
This is to train the model. 
```python
python main.py --mode=train 
```
## Bibtex
Please use the below BibTeX to cite if you use the code.
```
@ARTICLE{10605600,
  author={Uggi, Anuradha and Channappayya, Sumohana S.},
  journal={IEEE Signal Processing Letters}, 
  title={MS-NetVLAD: Multi-Scale NetVLAD for Visual Place Recognition}, 
  year={2024},
  volume={31},
  number={},
  pages={1855-1859},
  keywords={Visualization;Image recognition;Transforms;Contrastive learning;Benchmark testing;Feature extraction;Vectors;Image matching;visual place recognition;scale invariance;NetVLAD},
  doi={10.1109/LSP.2024.3425279}}
```
