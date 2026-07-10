# Material-Motif Network

This repository contains research code for constructing material-motif
bipartite networks and learning graph-based representations for materials
analysis and discovery.

## Overview

Crystal structures contain recurring local coordination environments that
can provide meaningful structural representations of materials.

The resulting network can be used to:

- Identify structurally and functionally related materials
- Generate material and motif embeddings
- Explore relationships across large materials databases
- Support downstream material-property prediction
- Compare motif-based representations with composition-based features

## Repository Structure

### Motif extraction

- `motif_extractor.py`: Extracts local coordination motifs and structural fingerprints.
- `motif_type_info.py`: Contains motif labels and coordination-environment definitions.

### BiNE embedding approach

- `graph.py` and `graph_utils.py`: Construct and process material-motif graphs.
- `data_utils.py`: Provides utilities for loading and preparing materials data.
- `train_neighbors.py`: Trains representations using graph-neighborhood information.
- `train_neighbors_explicit.py`: Uses explicit feature initialization.
- `train_neighbors_implicit.py`: Uses implicit embedding initialization.
- `data_split.ipynb`: Creates training and evaluation splits.

## Main Dependencies

- Python
- PyTorch
- NumPy
- pandas
- NetworkX
- scikit-learn
- pymatgen
- robocrys
