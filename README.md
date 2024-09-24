# Unsupervised Domain Adaptive Classification and Alignment (UDACA) for Hyperspectral Imagery

This repository implements the **Unsupervised Domain Adaptive Classification and Alignment (UDACA)** framework for hyperspectral image classification. The model is designed to improve classification accuracy in the target domain without the use of labeled target data. It aligns the feature distributions of the source and target domains using domain-level alignment techniques and style-perceive alignment (SPA).

## Overview

In many hyperspectral image (HSI) classification tasks, domain adaptation is critical when training data (source domain) and testing data (target domain) come from different distributions. This repository provides an implementation of a UDACA framework based on the following core components:
- **Feature Compaction Network**: Compresses high-dimensional spectral data into lower-dimensional feature representations.
- **Classifier**: Learns to classify features in the source domain.
- **Domain Discriminator**: Encourages feature distribution alignment between the source and target domains by distinguishing between them.
- **Style-Perceive Alignment (SPA)**: Aligns the style (statistics) of the source and target features using Gram matrices.

### Source Paper

The method implemented in this repository is based on the following paper:

> **Domain Adaptive Classification and Alignment Network for Cross-Scene Hyperspectral Image Interpretation**  
> DOI: [10.1109/TGRS.2021.3115432](https://ieeexplore.ieee.org/document/9606881)

### Dataset

The hyperspectral datasets used in this implementation are from the **Pavia University (PaviaU)** and **Pavia Center (PaviaC)** scenes. These datasets are available for download at the following link:

> [PaviaU and PaviaC Hyperspectral Datasets](https://drive.google.com/drive/folders/1XSSZcdp9fed4bxVrKKONNXbDzbYqTsjc?usp=sharing)

Both datasets contain ground-truth labels for training and testing. PaviaU has 103 spectral bands, and PaviaC has 102 spectral bands (after removing noise).

## Requirements

To run the code, you will need the following Python libraries:
- `torch`
- `numpy`
- `scipy`
- `scikit-learn`

You can install the required packages using the following command:

```bash
pip install torch numpy scipy scikit-learn
```

## Files

- **`udaca.py`**: Contains the implementation of the UDACA framework, including feature extraction, classifier, domain discriminator, style-perceive alignment, and training/evaluation functions.
- **`paviaU.mat`** and **`paviaC.mat`**: The original hyperspectral datasets (download and place in the `sample_data/` directory).
- **`paviaU_7gt.mat`** and **`paviaC_7gt.mat`**: Ground truth labels for classification (place in the `sample_data/` directory).

## Running the Model

### 1. Download the dataset

Download the datasets from the provided link and place them in the `sample_data/` folder. Ensure the structure looks like this:

```
sample_data/
  ├── paviaU.mat
  ├── paviaU_7gt.mat
  ├── paviaC.mat
  ├── paviaC_7gt.mat
```

### 2. Train the UDACA Model

To train the model with **PaviaU as the source domain** and **PaviaC as the target domain**, run the following:

```bash
python udaca.py
```

The script will:
1. Load the PaviaU and PaviaC datasets.
2. Train the UDACA model on the source domain (PaviaU).
3. Evaluate the model on the target domain (PaviaC).

### 3. Metrics

During evaluation, the script will output the following metrics for the target domain:
- **Overall Accuracy (OA)**
- **Average Accuracy (AA)**
- **Kappa Coefficient**

### 4. Modify Source and Target Domains

To train the model with **PaviaC as the source domain** and **PaviaU as the target domain**, modify the source argument in the `train_model()` function call:

```python
feature_net, classifier = train_model(paviaU_data, paviaU_gt, paviaC_data, paviaC_gt, source='paviaC')
```

## Model Components

- **FeatureCompactionNet**: Reduces the dimensionality of the hyperspectral data.
- **Classifier**: Classifies pixels based on the extracted features.
- **DomainDiscriminator**: Helps align the feature distributions between the source and target domains.
- **StylePerceiveLoss (SPA)**: Matches the Gram matrices of source and target features to ensure their statistical alignment.

## Citation

If you use this code or reference the methods, please cite the original paper:

```
@article{udaca2021,
  title={Domain Adaptive Classification and Alignment Network for Cross-Scene Hyperspectral Image Interpretation},
  author={Zhang, Wei and Gao, Lingjuan and Song, Jingying},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021},
  volume={60},
  pages={1-13},
  doi={10.1109/TGRS.2021.3115432}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
