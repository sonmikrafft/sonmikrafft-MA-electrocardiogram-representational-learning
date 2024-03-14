# Electrocardiogram and Representational Learning: Assessing Latent Factors and Similarity Measures

Code base for the Master's Thesis "Electrocardiogram and Representational Learning: Assessing Latent Factors and Similarity Measures" including the Interpretability Component, the training of a $\beta$-TCVAE on ECG data, and several tests of the Interpretability Component.

## The Interpretability Component

The Interpretability Component implements Similarity Measures and Disentanglement Metrics with the Strategy Pattern. 

## Test data

### dSprites

The metadata and latent representation z of dSprites is available by Locatello et al. (2019) “Challenging common assumptions in the unsupervised learning of disentangled
representations”
To run the tests, an example is provided in tests/dsprites_tests/data/dsprites0.csv

### ECG data

The previously stored ECG data is accessed with tfds 
- The synthetic dataset is created with Matlab and ECGSYN \[McSharry et al.(2003) "A dynamical model
for generating synthetic electrocardiogram signals"\]
- The real ECG dataset from Zheng et al.(2020) "A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients"


## Test model

For ECG data, one Pytorch model is provided in results/my_model/.

## Run Tests

Tests are realized as .ipynb.

- **src/train_vae.ipynb**: a model can be trained and evaluated on ECG data
- **tests/similarity_tests.ipynb**: tests all implemented Similarity Measures on a Tensorflow model and ECG data
- **tests/disentanglement_metrics_tests.ipynb**: tests all implemented Disentanglement Metrics on a Tensorflow model and ECG data
- **tests/dsprites_tests/similarity_tests_dis_lib.ipynb**: tests all implemented Disentanglement Metrics and associated Similarity Measures with dSprites



