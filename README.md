# Material-Motif Network

This repository contains research code for constructing material-motif bipartite networks and learning graph-based representations for materials analysis and discovery.

# Overview

Crystal structures contain recurring local coordination environments that can provide meaningful structural representations of materials. This project represents materials and their local structural motifs as two node types in a bipartite network. The resulting network can be used to:
-identify structurally and functionally related materials,
-generate material and motif embeddings,
-explore relationships across large materials databases,
-support downstream material-property prediction,
-compare motif-based representations with composition-based features.

# Repository Structure

motif_extractor.py: Extracts local coordination motifs and structural fingerprints from crystal structure files.
motif_type_info.py: Contains motif labels and coordination-environment definitions.
#BiNE approach to generate embeddings
graph.py and graph_utils.py :Construct and process material-motif graph representations.
data_utils.py :Provides utilities for loading and preparing materials data.
train_neighbors.py: Trains material and motif representations using graph-neighborhood information.
train_neighbors_explicit.py: Trains representations using explicit feature initialization.
train_neighbors_implicit.py: Trains representations using implicit embedding initialization.
data_split.ipynb: Creates training and evaluation data splits.

# Main Dependencies

Python
PyTorch
NumPy
pandas
NetworkX
scikit-learn
pymatgen
robocrys
