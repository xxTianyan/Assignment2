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
# Create a new conda environment and activate it
conda create -n msa python=3.12
conda activate msa
```
Ensure that pip is using the most up to date version of setuptools:
```sh
pip install --upgrade pip setuptools wheel
```
Create an editable install of the bisection method code (note: you must be in the correct directory):
```sh
pip install -e .
```


## Main Classes
1. **Define Nodes:** Create and add nodes with unique IDs and coordinates.
3. **Define Sections:** Assign cross-sectional properties.
4. **Define Elements:** Connect nodes using beam elements with assigned material and section.


### Example
See examples folder for details.

