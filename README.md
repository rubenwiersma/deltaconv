# DeltaConv
[[Paper]](https://rubenwiersma.nl/assets/pdf/DeltaConv.pdf) [[Project page]](https://rubenwiersma.nl/deltaconv) [[Replicability Stamp]](http://www.replicabilitystamp.org#https-github-com-rubenwiersma-deltaconv)

Code for the SIGGRAPH 2022 paper "[DeltaConv: Anisotropic Operators for Geometric Deep Learning on Point Clouds](https://arxiv.org/abs/2111.08799)" by Ruben Wiersma, Ahmad Nasikun, Elmar Eisemann, and Klaus Hildebrandt.

Anisotropic convolution is a central building block of CNNs but challenging to transfer to surfaces. DeltaConv learns combinations and compositions of operators from vector calculus, which are a natural fit for curved surfaces. The result is a simple and robust anisotropic convolution operator for point clouds with state-of-the-art results.

![](img/deltaconv.png)

*Top: unlike images, surfaces have no global coordinate system. Bottom: DeltaConv learns both scalar and vector features using geometric operators.* 

## Contents
- [Installation](#installation)
- [Replicating the experiments](#replicating-the-experiments)
- [Tests](#tests)
- [Citation](#citations)

## Installation
1. Clone this repository:
```bash
git clone https://github.com/rubenwiersma/deltaconv.git
```

2. Create a conda environment from the `environment.yml`:
```bash
conda env create -n deltaconv -f environment.yml
```

Done!

### Manual installation
If you wish to install DeltaConv in your own environment, proceed as follows.

1. Make sure that you have installed:
    - Numpy - `pip install numpy`
    - [PyTorch](https://pytorch.org/get-started/locally/) - see [instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
    - [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) - `conda install pyg -c pyg`

2. Install DeltaConv:
```bash
pip install deltaconv
```
### Building DeltaConv for yourself
1. Make sure you clone the repository with submodules:
```bash
git clone --recurse-submodules https://github.com/rubenwiersma/deltaconv.git
```
If you have already cloned the repository without submodules, you can fix it with `git submodule update --init --recursive`.

2. Install from folder:
```bash
cd [root_folder]
pip install
```


## Replicating the experiments
See the README.md in `replication_scripts` for instructions on replicating the experiments and using the pre-trained weights (available in `experiments/pretrained_weights`).

In short, you can run bash scripts to replicate our experiments. For example, evaluating pre-trained weights on ShapeNet:
```bash
cd [root_folder]
conda activate deltaconv
bash replication_scripts/pretrained/shapenet.sh
```

You can also directly run the python files in `experiments`:
```bash
python experiments/train_shapenet.py
```
Use the `-h` or `--help` flag to find out which arguments can be passed to the training script:
```bash
python experiments/train_shapenet.py -h
```

You can keep track of the training process with tensorboard:
```bash
tensorboard logdir=experiments/runs/shapenet_all
```

### Anisotropic Diffusion
The code that was used to generate Figure 2 from the paper and Figure 2 and 3 from the supplement is a notebook in the folder `experiments/anisotropic_diffusion`.

## Data
ModelNet40, ShapeNet, SHREC, and human body shape segmentation automatically download the datasets from a public repository and place them in the correct folder. The data for ScanObjectNN can be downloaded from the ScanObjectNN website: https://hkust-vgd.github.io/scanobjectnn/. Download and extract the files into `experiments/data/ScanObjectNN/raw`. The folder structure in `experiments/data/ScanObjectNN` should look like:
```
ScanObjectNN
└─── raw
     └─── main_split
     |    | train_objectdataset.h5
     |    | test_objectdataset.h5
     |    | ...
     |
     └─── main_split_nobg
          | train_objectdataset.h5
          | test_objectdataset.h5
          | ...
```

## Tests
In the paper, we make statements about a number of properties of DeltaConv that are either a result of prior work or due to the implementation. We created a test suite to ensure that these properties hold for the implementation, along with unit tests for each module. For example:
- Section 3.6, 3.7: Vector MLPs are equivariant to norm-preserving transformations, or coordinate-independent (rotations, reflections)
    - `test/nn/test_mlp.py`
    - `test/nn/test_nonlin.py`
- Section 3.7: DeltaConv is coordinate-independent, a forward pass on a shape with one choice of bases leads to the same output and weight updates when run with different bases
    - `test/nn/test_deltaconv.py`
- Introduction, section 3.2: The operators are robust to noise and outliers.
    - `test/geometry/test_grad_div.py`
- Supplement, section 1: Vectors can be mapped between points with equation (15).
    - `test/geometry/test_grad_div.py`

## Citations
Please cite our paper if this code contributes to an academic publication:

```bib
@Article{Wiersma2022DeltaConv,
  author    = {Ruben Wiersma, Ahmad Nasikun, Elmar Eisemann, Klaus Hildebrandt},
  journal   = {Transactions on Graphics},
  title     = {DeltaConv: Anisotropic Operators for Geometric Deep Learning on Point Clouds},
  year      = {2022},
  month     = jul,
  number    = {4},
  volume    = {41},
  doi       = {10.1145/3528223.3530166},
  publisher = {ACM},
}
```

The farthest point sampling code relies on Geometry Central:
```bib
@misc{geometrycentral,
  title = {geometry-central},
  author = {Nicholas Sharp and Keenan Crane and others},
  note = {www.geometry-central.net},
  year = {2019}
}
```

And we make use of PyG (and underlying packages) to load point clouds, compute sparse matrix products, and compute nearest neighbors:
```bib
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
```

