# GNN-MGDeform: Metallic Glass Deformation Prediction

This repository contains the training pipeline for predicting the shaping and deformation of Metallic Glass (MG) using Graph Neural Networks (GNN). The code is designed to be easy to run, with built-in visual tracking and automatic model saving.

## About the Dataset

The graph data is already prepared and located in the `MGDataset` directory. As the file size exceeds the GitHub limit, it has been uploaded to the Figshare shared repository with the link: . You only need to simply place this file in the directory like this: "./MGDataset".

**Sample Details:**
* The primary dataset consists of 120 Cu64Zr36 metallic glass samples.
* Each sample contains exactly 6,480 atoms.
* The samples were generated at a cooling rate of 10^10 K/s.

**Simulation Process:**
* All simulations were performed using LAMMPS, with Periodic Boundary Conditions (PBC) applied in all three spatial directions.
* For each MG sample, twelve fundamental athermal quasi-static (AQS) loading modes are applied. These include uniaxial tension and compression along the x, y, and z axes, as well as simple shear in the xy±, yz±, and xz± directions.
* During each deformation step, a small affine strain of 10^-4 is imposed. This is immediately followed by structural relaxation using conjugate gradient energy minimization. 
* Because any complex loading scenario can be viewed as a linear combination of these twelve elementary modes, they provide a highly comprehensive measure of local plastic resistance.

**Target Prediction Variable:**
* To measure the atomic propensity for plasticity, we evaluate the non-affine displacement squared (D²) at applied strains of 10%.
* Raw D² values naturally show a lognormal distribution (they are highly skewed). To make neural network training more stable and efficient, we use its natural logarithm, ln(D²).
* Finally, these ln(D²) values are averaged across all twelve fundamental loading modes and further normalized using the empirical cumulative distribution function, denoted as ECDF(ln(D²)).

## Key Features

* **Automatic Data Splitting**: The script forces a random split of your dataset into exactly 100 training samples, 10 validation samples, and 10 test samples.
* **Smart Training**: It uses the RAdam optimizer paired with a Cosine Annealing learning rate scheduler to help the model learn smoothly.
* **Early Stopping**: To prevent overfitting, training will automatically stop if the Validation Pearson Correlation Coefficient (PCC) does not improve for 50 epochs.
* **Auto-Visualization**: The script automatically generates scatter plots and loss curves during training so you can visually track how well the model is performing.
* **Organized Outputs**: Every time you run a training session, it creates a new timestamped folder in the `Output` directory. This folder safely stores your copied config file, normalizer parameters, best model weights, and graphs.

## Getting Started

### 1. Adjust Configurations
All main settings are managed in `Config.yaml`. Here are some of the key default settings:
* `batch_size`: 1
* `total_epochs`: 1000
* `lr` (Learning Rate): 0.00015
* `hidden_dim`: 128

### 2. Run the Training
Start the training process by simply running:
```bash
python Training.py
