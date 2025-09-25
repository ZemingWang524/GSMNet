# GSMNet 


## Installation
```
# Clone the repository
git clone https://github.com/ZemingWang524/GSMNet.git
cd GSMNet

# Create a Conda environment (original env)
conda env create -f environment.yml

# Activate the environment
conda activate GSMNet-env
```

## Dependencies
The environment used for the results reported in the paper relies on these dependencies:
```
torch==2.4.0
torch-scatter==2.1.2
pytorch-ignite==0.5.2
scikit-learn==1.6.1
scipy==1.13.1
pandas==2.2.3
yacs==0.1.8
jarvis-tools==2022.9.16 # Note that this version may be related to reproducibility
numpy==1.26.4
matplotlib=3.10.3
dgl==2.4.0
periodictable==2.0.2
pydantic==2.11.4
opt-einsum==3.4.0
tensorboard==2.19.0
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

## Pre-trained Models
Links to download pre-trained models:

- [google drive](https://drive.google.com/file/d/1nEjVhv1rD8KWVDqVbkRiVyEbSePHUF4V/view?usp=sharing)

## Dataset

### JARVIS Dataset
For JARVIS Dataset, we follow MatFormer and use the same training, validation, and test set. We evaluate our PMCGNN on five important crystal property tasks, including formation energy, bandgap(OPT), Total energy, Bandgap(MBJ), and Ehull. The training, validation, and test set contains 44578, 5572, and 5572 crystals for tasks of Formation Energy, Total Energy, and Bandgap(OPT). The numbers are 44296, 5537, 5537 for Ehull, and 14537, 1817, 1817 for Bandgap(MBJ). The used metric is test MAE. The baseline results are taken from PMCGNN and ComFormer, while other results are from the authors' manual reproduction in a unified environment.

### The Materials Project Dataset
For The Materials Project Dataset, we follow MatFormer and use the same training, validation, and test set. We evaluate our PMCGNN on four important crystal property tasks, including Formation Energy, Band Gap, Bulk Moduli and Shear Moduli. The training, validation, and test set contains 60000, 5000, and 4239 crystals for tasks of formation energy and band gap. The numbers are 4664, 393, 393 for Bulk Moduli and Shear Moduli. The used metric is test MAE. The baseline results are taken from PMCGNN and ComFormer, while other results are from the authors' manual reproduction in a unified environment.

## Results

### The Materials Project

Results on MP Dataset:
| Method         | Formation Energy ↓ | Bandgap ↓ | Bulk Moduli ↓ | Shear Moduli ↓ |
|----------------|-------------------|-----------|---------------|---------------|
| CGCNN          | 0.031             | 0.292     | 0.047         | 0.077         |
| SchNet         | 0.033             | 0.345     | 0.066         | 0.099         |
| MEGNet         | 0.030             | 0.307     | 0.051         | 0.099         |
| GATGNN         | 0.033             | 0.280     | 0.045         | 0.075         |
| ALIGNN         | 0.0221            | 0.218     | 0.051         | 0.078         |
| Matformer      | 0.0210            | 0.211     | 0.043         | 0.073         |
| PotNet         | 0.0188            | 0.204     | 0.040         | 0.065       |
| eComFormer     | 0.01816         | 0.202     | 0.0417        | 0.0729        |
| iComFormer     | 0.01826           | 0.193   | 0.038       | 0.0637        |
| PMCGNN         | _0.0170_          | _0.186_   | 0.038         | _0.063_       |
| CartNet        | 0.0175            | 0.188     | **0.033**     | 0.0645        |
| GSMNet | **0.0168**     | **0.182** | _0.034_       | **0.0618**    |


(best result in **bold** and second best in _italic_)

### Jarvis Dataset

Results on Jarvis Dataset:
| Method         | Form. Energy ↓ | Band Gap (OPT) ↓ | Total Energy ↓ | Band Gap (MBJ) ↓ | Ehull ↓ |
|----------------|----------------|------------------|----------------|------------------|---------|
| CGCNN          | 0.0630         | 0.200            | 0.078          | 0.410            | 0.170   |
| SchNet         | 0.0450         | 0.190            | 0.047          | 0.430            | 0.140   |
| MEGNet         | 0.0470         | 0.145            | 0.058          | 0.340            | 0.084   |
| GATGNN         | 0.0470         | 0.170            | 0.056          | 0.510            | 0.120   |
| ALIGNN         | 0.0331         | 0.142            | 0.037          | 0.310            | 0.076   |
| Matformer      | 0.0325         | 0.137            | 0.035          | 0.300            | 0.064   |
| PotNet         | 0.0294         | 0.127            | 0.032          | 0.270            | 0.055   |
| eComFormer     | 0.0284         | 0.124            | 0.032          | 0.280            | _0.044_ |
| iComFormer     | _0.0272_       | _0.122_            | 0.0288       | 0.260            | 0.047   |
| PMCGNN         | 0.0278         | _0.122_            | 0.029          | _0.250_          | **0.040** |
| CartNet        | 0.0278         | _0.122_            | **0.0264**     | 0.252            | _0.044_ |
| GSMNet | **0.0271**  | **0.120**        | _0.0282_       | **0.227**        | **0.040** |


(best result in **bold** and second best in _italic_)

## Known Issues
Due to the presence of certain non-deterministic operations in PyTorch, some results may not be exactly reproducible and may exhibit slight variations. This variability can also arise when using different GPU models for training and testing the network.

## Acknowledgement
This repo is built upon the previous work PMCGNN's [[codebase]](https://github.com/yinhexingxing/PMCGNN/tree/main), licensed under the GPL-3.0 license.  
This repo is partially based on GRIT's [[codebase]](https://github.com/LiamMa/GRIT/tree/main) and Mamba's [[codebase]](https://github.com/state-spaces/mamba).

Thank you very much for these excellent codebases. 
