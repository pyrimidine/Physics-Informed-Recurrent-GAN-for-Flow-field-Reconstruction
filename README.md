

# Physics-Informed Recurrent GAN for Flow Field Reconstruction

This code implements a physics-informed recurrent super-resolution generative reconstruction model for use in rotating detonation combustors.

## Dependencies
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset is generated using the in-house code TurfSIM of SCP and is approximately 50GB in size.

### Downloading Data
1. The source files are listed in the **source** directory. You can download the .dat source files from the following link: [Download Source Files](https://cloud.tsinghua.edu.cn/d/5b1a402e5a1549d0a3a5/).

2. After downloading the .tif files, run **generate_data.py** located in the **source** directory. This script will process the source files and construct an index for the dataset.

## Example Usage

To initiate distributed training with a batch size of 64 for 200 epochs, use the following command:
```bash
python train.py --batchSize 64 --nEpochs 200
```

For a complete evaluation of the test set, use:
```bash
python test.py --dic #path_to_model
```

## Citation
If you find this work useful, please cite it as follows:

```bibtex
@article{Wang2024,
    title = {Physics-informed recurrent super-resolution generative reconstruction in rotating detonation combustor},
    journal = {Proceedings of the Combustion Institute},
    volume = {40},
    number = {1},
    pages = {105649},
    year = {2024},
    issn = {1540-7489},
    doi = {https://doi.org/10.1016/j.proci.2024.105649},
    author = {Xutun Wang and Haocheng Wen and Quan Wen and Bing Wang},
    keywords = {Flow-field reconstruction, Physics-informed machine learning, Generative adversarial network, Recurrent neural network, Rotating detonation combustor}
}
