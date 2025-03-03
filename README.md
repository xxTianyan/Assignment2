# Direct Stiffness Method for Structural Analysis

## Overview
This Python implementation applies the **Direct Stiffness Method** to analyze **3D beam structures**. It models nodes, materials, sections, and elements, computes local and global stiffness matrices, applies boundary conditions, and solves for displacements and reaction forces.

## Installation
Firstly, clone or download the repository to your local environment:
```sh
# Clone the repository
git clone https://github.com/xxTianyan/Assignment2.git
cd Assignment2
```
Create a virtual environment, here we use conda:
```sh
# Create a new conda environment, activate it and install python
conda create -n msa
conda activate msa
conda install python
```
Ensure that pip is using the most up to date version of setuptools:
```sh
pip install --upgrade pip setuptools wheel
```
Create an editable install of the msa module (note: you must be in the correct directory):
```sh
pip install -e .
```


## Main Process
1. **Define Nodes:** Create and add nodes with unique IDs, coordinates and loads. Specify boundary conditions.
2. **Define Elements:** Connect nodes using beam elements with assigned material and section.
3. **Assemble Structure:** Construct and analyze the target structure,  build global stiffness and force matrices.
4. **Solve Results:** Solve for Displacements: Compute nodal displacements. Compute Reactions: Calculate reaction forces.


### Example
See examples folder for details. But I failed to finish the computing elastic critical load codes.

