# Revisiting Deep Feature Reconstruction for Logical and Structural Industrial Anomaly Detection

This is the official repository for our paper "Revisiting Deep Feature Reconstruction for Logical and Structural Industrial Anomaly Detection"

> **Revisiting Deep Feature Reconstruction for Logical and Structural Industrial Anomaly Detection**
> Anonymous Authors
>
> [[`Paper`](README.md)]

Industrial anomaly detection is crucial for quality control and predictive maintenance but is challenging due to limited training data, varied anomaly types, and changing external factors affecting object appearances. Existing methods detect structural anomalies, such as dents and scratches, by relying on multi-scale features of image patches extracted from a deep pre-trained network. Nonetheless, extensive memory or computing requirement hinders their adoption in practice. Furthermore, detecting logical anomalies, such as images with missing or surplus elements, necessitates understanding spatial relationships beyond traditional patch-based methods. Our work focuses on Deep Feature Reconstruction (DFR), which offers a memory- and compute-efficient way of detecting structural anomalies. Moreover, we extend DFR to develop a unified framework for detecting structural and logical anomalies, called ULSAD. Specifically, we improve the training objective of DFR to enhance the capability to detect structural anomalies and introduce an attention-based loss using a global autoencoder-like network for detecting logical anomalies. Empirical results on five benchmark datasets demonstrate the effectiveness of ULSAD in the detection and localization of both structural and logical anomalies compared to eight state-of-the-art approaches. Moreover, an in-depth ablation study showcases the importance of each component in enhancing overall performance.

## Installation
This code is written in `Python 3.9` and requires the packages listed in [`requirements.txt`](requirements.txt).

To run the code, set up a virtual environment using `conda`:

```
cd <path-to-cloned-directory>

conda create -n anomalib_env python=3.10
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```

## Running experiments

To run an experiment create a new configuration file in the [`ulsad`](anomalib/models/ulsad) directory. The experiments can be can run using the following command:

```
cd <path-to-cloned-directory>
python  main.py --config anomalib/models/ulsad/config/<config-file-name>.yaml
```

We provide the configuration files used for our experiments.


## Remarks

Our code is based on the open-source project [Anomalib](https://github.com/openvinotoolkit/anomalib). We convey our gratitude to the developers.