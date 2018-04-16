# Using deep neural nets for classification of diffraction images

## Description

This is the code for the paper _Using deep neural nets for classification of diffraction images_, Zimmermann et al. 2018, doi: 000


## Abstract

Intense short-wavelength pulses from Free-Electron-Lasers (FELs) and high-harmonic-generation (HHG) sources enable diffractive imaging of individual nano-sized objects with a single x-ray laser shot. Due to the high repetition rates, large data sets with up to several million diffraction patterns are typically obtained in FEL particle diffraction experiments, representing a severe problem for data analysis. Assuming a dataset of K diffraction patterns with M x N pixels, a high dimensional space (K x M x N) has to be analyzed. Thus feature selection is crucial as it reduces the dimensionality. This is typically achieved via custom-made algorithms that do not generalize well, e.g. feature extraction methods applicable to spherically shaped patterns but not to arbitrary shapes. This work exploits the possibility to utilize a deep neural network (DNN) as a feature extractor.
% A workflow scheme is proposed based on a residual convolutional neural net (ResNet), that drastically reduces the amount of work needed for the classification of large diffraction datasets, only a small fraction of the data has to be manually classified.
A residual convolutional DNN (ResNet) is trained in a supervised manner utilizing a small training set of manually labeled data.
In addition to that we enhance the classification accuracy by proposing a novel activation function, which takes the intrinsic scaling of diffraction pattern into account, and benchmark two widely used DNN architectures. Furthermore, an approach to be more robust to highly noisy data using two-point cross correlation maps is presented. We conduct all experiments on two large datasets, first, data from a wide-angle X-ray scattering (WAXS) experiment on Helium nanodroplets, conducted at the LDM endstation of the FERMI free-electron laser in Trieste, and second, a small-angle X-ray scattering (SAXS) dataset from the CXI-database (CXIDB) , provided as part of an ongoing effort to advance the programmatic and algorithmic description of diffraction images .
