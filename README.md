# COME
Code for paper: [Learning Compact Semantic Information for Incomplete Multi-View Missing Multi-Label Classification](https://icml.cc/virtual/2025/poster/45273)
Pascal07 data is prepared for a demo, you can download data from [here](https://drive.google.com/drive/folders/1cixudQjQ1I1pvRFy0Srsl2UTgvzv-o2k?usp=drive_link).
You can run 'python main.py' for a demo!

## Abstract
Multi-view data involves various data forms, such as multi-feature, multi-sequence and multimodal data, providing rich semantic information for downstream tasks. The inherent challenge of incomplete multi-view missing multi-label learning lies in how to effectively utilize limited supervision and insufficient data to learn discriminative representation. Starting from the sufficiency of multi-view shared information for downstream tasks, we argue that the existing contrastive learning paradigms on missing multi-view data show limited consistency representation learning ability, leading to the bottleneck in extracting multi-view shared information. In response, we propose to minimize task-independent redundant information by pursuing the maximization of cross-view mutual information. Additionally, to alleviate the hindrance caused by missing labels, we develop a dual-branch soft pseudo-label cross-imputation strategy to improve classification performance. Extensive experiments on multiple benchmarks validate our advantages and demonstrate strong compatibility with both missing and complete data.

## Overview of COME
<img width="4187" height="1787" alt="architecture" src="https://github.com/user-attachments/assets/52eea7c5-b830-48d8-b587-94c56c20f2d8" />

## Environment
Please run the following command in the shell, as specified in `requirements.txt`:
```bash
conda create -n COME
conda activate COME
pip install -r requirements.txt
```
## Citation
If you find this work useful, please consider citing it:
```
@inproceedings{wen2025learning,
  title={Learning Compact Semantic Information for Incomplete Multi-View Missing Multi-Label Classification},
  author={Wen, Jie and Liu, Yadong and Tang, Zhanyan and He, Yuting and Chen, Yulong and Li, Mu and Liu, Chengliang},
  booktitle={International Conference on Machine Learning},
  pages={66467--66480},
  year={2025},
  organization={PMLR}
}
```
