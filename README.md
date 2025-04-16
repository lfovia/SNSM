# SNSM
Source Code for Self-Neighbourhood Support Maps Project
This repository provides the implementation of the [SNSM](https://ieeexplore.ieee.org/abstract/document/10889993) approach. Some of the code snippets are stolen from [AnyLoc](https://github.com/AnyLoc/AnyLoc.git) and [Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD.git) works.  

## Summary
This work proposes a new training-free technique for the all-day Visual Place Recognition (VPR) problem. It especially addresses the illumination challenge in the VPR. The SNSM function accepts a feature block from the backbone model and aggregates it into a modality-invariant (RGB and thermal) feature map called an SNSM map. Essentially, this attempts to capture the support value of each selected patch from its neighbourhood, which retains the homogeneous structural details and suppresses the heterogeneous modality-specific features. For further details, please refer to the full paper. The simple and training-free SNSM improve upon popular VPR models and various unsupervised methods by a considerable margin.       

## Repo. details
The SNSM directory contains the All-Backbones-VLAD_bl.py file, which can produce recall rates for the choice of dataset and aggregator. Currently, only unsupervised feature extraction techniques are included. However, VPR models reported in the paper are off-the-shelf, and models are open-source. 
## Data
Inference datasets: [RGB-T Datasets Drive link](https://drive.google.com/file/d/11qBQw9DadQ5MemUTouK6HolPEPCo-BXT/view?usp=sharing) 

## Command to run
Run the below command for inference. 
```python
sh All-Backbones-VLAD_bl.sh 
```
snsm aggregator is activated by default. Please provide appropriate arguments in the bash file regarding the choice of aggregator. The available aggregators include VLAD, VLAD-API, GeM, GAP, GMP, and SNSM. More information about these is available in the main function in All-Backbones-VLAD_bl.py.  
## Bibtex
Please use the below BibTeX to cite if you use the code.
```
@INPROCEEDINGS{10889993,
  author={Uggi, Anuradha and Channappayya, Sumohana},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Training-free Adapter for Multi-Modal Image Matching for All-Day Visual Place Recognition}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Computer vision;Adaptation models;Image recognition;Correlation;Source coding;Speech recognition;Signal processing;Acoustics;Speech processing;Visual place recognition;Multi-modal image retrieval;RGB;thermal;and visual place recognition},
  doi={10.1109/ICASSP49660.2025.10889993}}
```
