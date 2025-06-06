# Joint Clustering Model for Text-Attributed Networks

This repository contains the code developed for my final-year **M4R** research project in Bayesian inference for latent structure models. The project focuses on a joint clustering framework for text-attributed networks, combining spectral embeddings of graph structure with node-level textual features.

The model integrates Gaussian likelihoods over graph embeddings with Dirichlet-Multinomial models for text, and performs inference via collapsed Gibbs sampling. It includes support for modality reweighting, and evaluation on both simulated data and the Cora citation network.