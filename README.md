# Physics-Informed-Recurrent-GAN-for-Flow-field-Reconstruction


This code was used to implement: Physics-informed Recurrent Super-Resolution Generative Reconstruction in Rotating Detonation Combustor (2023)



## Dependencies
```
pip install -r requirements.txt
```

## Dataset

The dataset is calculated by in-house code TurfSIM of SCP around ~50GB.

### Downloading data
1. The source files are listed in the **source** directory. Open the following link to download .dat source files: https://cloud.tsinghua.edu.cn/d/5b1a402e5a1549d0a3a5/

2. After the tif files are downloaded, run **generate_data.py** (under **source** directory). This will process the source files and construct an index for the dataset.

## Example

For distributed training with batch size of 64, 200 epochs:
```
python train.py --batchSize 64 --nEpochs 200
```
For a complete test set evaluation:
```
python test.py --dic #path_to_model
```

## Cite
title = {Physics-informed recurrent super-resolution generative reconstruction in rotating detonation combustor},

journal = {Proceedings of the Combustion Institute},

volume = {40},

number = {1},

pages = {105649},

year = {2024},

issn = {1540-7489},

doi = {https://doi.org/10.1016/j.proci.2024.105649},

author = {Xutun Wang and Haocheng Wen and Quan Wen and Bing Wang},

keywords = {Flow-field reconstruction, Physics-informed machine learning, Generative adversarial network, Recurrent neural network, Rotating detonation combustor},
