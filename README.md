# Classifying cell groups of interest given single cell mass spectra through interpretable machine learning
The code repository for results in the paper: 
[Single-Cell Classification Using Mass Spectrometry Through Interpretable Machine Learning (Xie el al, Anal.Chem. 2020)](https://pubs.acs.org/doi/10.1021/acs.analchem.0c01660)

<p align="center">
<img src="https://github.com/richardxie1119/SCCML/blob/master/figure/main.png" width="500",align="middle">
</p>


### Dependencies:
- **NumPy**

- **scikit-learn**: https://github.com/scikit-learn/scikit-learn

- **SHAP**: https://github.com/slundberg/shap


### To replicate results in paper
Code to reproduce the figures in the paper are available in [this notebook](https://github.com/richardxie1119/SCCML/blob/master/notebooks/sc_classification.ipynb)

Three data sets are available and used to demonstrate the classification workflow:
- **HIP_CER.pkl**: 1201 single cells from different rat brain regions (hippocampus or cerebellum, m/z 50-500) on a 7T Fourier transform ion cyclotron resonance (FT-ICR) mass spectrometer.

- **ICC_rms.pkl**: 1544 rat cerebellar single cells labeled by immunocytochemistry (neurons or astrocytes, m/z 500-1000) acquired using MALDI-time-of-flight (TOF)-MS

- **SIMS.pkl**: 1542 single cells from different rat brain regions (dorsal root ganglia (DRG) or cerebellum, m/z 500-850) acquired using secondary ion mass spectrometry

Download this repo containing the notebooks and the scripts, and download the [processed data sets](https://drive.google.com/drive/folders/1a__5nrhcNX8ePqD31NZlsYlWKyIZaT8W?usp=sharing). Place the data sets into a data folder, and run the code in sc_classification notebook.


