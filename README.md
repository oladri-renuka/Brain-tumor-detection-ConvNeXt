# Brain Tumor Classification using ConvNext Model

This repository implements a deep learning model based on the ConvNext architecture for classifying brain tumors from MRI scans. The model is built using the PyTorch framework and leverages transfer learning with the ConvNext architecture pretrained on ImageNet.

## Project Overview

The goal of this project is to classify brain tumors into 42 categories based on MRI scans. We use a ConvNext model, fine-tuned for this task, and evaluate its performance on a dataset of brain MRI images.

## Dataset

The dataset consists of brain MRI images categorized into the following classes:

- **Astrocitoma**: T1, T1C+, T2
- **Carcinoma**: T1, T1C+, T2
- **Ependimoma**: T1, T1C+, T2
- **Ganglioglioma**: T1, T1C+, T2
- **Germinoma**: T1, T1C+, T2
- **Glioblastoma**: T1, T1C+, T2
- **Granuloma**: T1, T1C+, T2
- **Meduloblastoma**: T1, T1C+, T2
- **Meningioma**: T1, T1C+, T2
- **Neurocitoma**: T1, T1C+, T2
- **Oligodendroglioma**: T1, T1C+, T2
- **Papiloma**: T1, T1C+, T2
- **Schwannoma**: T1, T1C+, T2
- **Tuberculoma**: T1, T1C+, T2
- **Normal**: T1, T2

## Training

The model is trained using the ConvNext architecture with the following hyperparameters:

- **Batch size**: 32
- **Learning rate**: 1e-4
- **Epochs**: 15
- **Optimizer**: Adam
- **Loss function**: CrossEntropyLoss

### Data Augmentation

The following transformations are applied to the training data:

- Resize to 224x224
- Random Rotation between -25 and 20 degrees
- Normalize using ImageNet mean and standard deviation

## Evaluation

The model is evaluated using the validation set. Metrics include:

- **Accuracy**
- **Classification Report**
- **ROC AUC Score**

### Results

- **Accuracy Score**: 0.9964

## Model Architecture

The model is based on the ConvNext Tiny architecture, with modifications for the specific brain tumor classification task. The final model includes:

- A base ConvNext feature extractor
- A custom fully connected layer for classification

### Comparison with EfficientNetB5

The results are compared with the state-of-the-art model EfficientNetB5, which includes:

- **Base model**: EfficientNetB5 pretrained on ImageNet
- Batch Normalization layer for improved training stability
- Dense layers for classification
- Dropout layer to prevent overfitting
- L2 and L1 regularization to enhance generalization

- **EfficientNetB5 Test Accuracy**: 0.9286
