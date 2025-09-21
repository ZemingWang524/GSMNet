# GSMNet 


## Environment Setup
```
# Create a Conda environment (original env)
conda env create -f environment.yml

# Activate the environment
conda activate GSMNet-env
```
## Summation Algorithm
We used the summation algorithm from [PotNet](https://github.com/divelab/AIRS/tree/main/OpenMat/PotNet).
## Train Models
```
python main.py --config configs/gsmnet.yaml
```
## Evaluate Models
```
python main.py --config configs/gsmnet.yaml  --checkpoint xxx --testing
```
We provide pretrained models in this [google drive](https://drive.google.com/file/d/1nEjVhv1rD8KWVDqVbkRiVyEbSePHUF4V/view?usp=sharing). 

## Dataset

### JARVIS Dataset
For JARVIS Dataset, we follow MatFormer and use the same training, validation, and test set. We evaluate our PMCGNN on five important crystal property tasks, including formation energy, bandgap(OPT), Total energy, Bandgap(MBJ), and Ehull. The training, validation, and test set contains 44578, 5572, and 5572 crystals for tasks of Formation Energy, Total Energy, and Bandgap(OPT). The numbers are 44296, 5537, 5537 for Ehull, and 14537, 1817, 1817 for Bandgap(MBJ). The used metric is test MAE. The baseline results are taken from PMCGNN and ComFormer.

### The Materials Project Dataset
For The Materials Project Dataset, we follow MatFormer and use the same training, validation, and test set. We evaluate our PMCGNN on four important crystal property tasks, including Formation Energy, Band Gap, Bulk Moduli and Shear Moduli. The training, validation, and test set contains 60000, 5000, and 4239 crystals for tasks of formation energy and band gap. The numbers are 4664, 393, 393 for Bulk Moduli and Shear Moduli. The used metric is test MAE. The baseline results are taken from PMCGNN and ComFormer.

## Known Issues
Due to the presence of certain non-deterministic operations in PyTorch, some results may not be exactly reproducible and may exhibit slight variations. This variability can also arise when using different GPU models for training and testing the network.

## Acknowledgement
This repo is built upon the previous work PMCGNN's [[codebase]](https://github.com/yinhexingxing/PMCGNN/tree/main), licensed under the GPL-3.0 license.  
This repo is partially based on GRIT's [[codebase]](https://github.com/LiamMa/GRIT/tree/main)

Thank you very much for these excellent codebases. 
