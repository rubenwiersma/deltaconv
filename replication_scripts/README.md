# DeltaConv Replication

Use the scripts in this folder to replicate the experiments in the paper. Run the scripts while you're in the root folder of this repository and make sure to activate the `deltaconv` environment.

Example usage:
```bash
cd [root_folder]
conda activate deltaconv
bash replication_scripts/shapenet.sh
```

## Training from scratch

### Paper
- Table 1: `modelnet40.sh`
- Table 2: `scanobjectnn.sh` - See download instructions in the main readme
- Table 3: `shrec.sh` - Average the test results afterward
- Table 4: `shapenet.sh`

### Supplement
- Table 1: `shapeseg.sh`
- Table 2: `shapenet.sh`

## Evaluating pre-trained weights

### Paper
- Table 1: `pretrained/modelnet40.sh`
- Table 2: `pretrained/scanobjectnn.sh` - See download instructions in the main readme
- Table 3: `pretrained/shrec.sh` - Average the test results afterward
- Table 4: `pretrained/shapenet.sh`

### Supplement
- Table 1: `pretrained/shapeseg.sh`
- Table 2: `pretrained/shapenet.sh`

## Anisotropic Diffusion
An interactive notebook of the anisotropic diffusion experiments can be found in `experiments/anisotropic_diffusion`.

