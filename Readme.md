# Signature Verification Using Siamese Neural Networks

A thesis work that implements a **Siamese neural network** for offline handwritten signature verification. The model learns to distinguish genuine from forged signatures using a distance-based metric-learning approach built on a modified Xception architecture.

## Overview

This project addresses the problem of **offline signature verification** — determining whether a given handwritten signature is genuine or a forgery. It uses a Siamese network architecture that learns an embedding space where genuine signature pairs are close together, and genuine-forgery pairs are far apart.

The system is trained and evaluated on the **CEDAR signature dataset**, which contains 55 signers with 24 genuine and 24 forged signatures each (1,320 genuine + 1,320 forged samples).

## Architecture

- **Backbone**: Modified Xception network with entry, middle, and exit flows using depthwise separable convolutions
- **Embedding**: Dense layer projecting features into a compact embedding space
- **Distance Layer**: Custom Euclidean distance layer with ReLU activation (capped at alpha) to measure similarity between signature pairs
- **Training Strategy**: Contrastive learning with pairwise matching — genuine-genuine pairs (must-link) and genuine-forgery pairs (cannot-link)

## Key Features

- **Image Preprocessing**: Background removal, contrast/brightness/sharpness adjustment, grayscale conversion, and inversion for cleaner signature images
- **Data Augmentation**: Shift, scale, rotation, flip, optical/grid distortion, and cutout via the `albumentations` library
- **Custom Data Generator**: `AEGenerator` class (extends `tf.keras.utils.Sequence`) that generates balanced pairs of genuine-genuine and genuine-forgery signatures per batch
- **Train/Test Split**: First 45 signers for training, remaining 10 for testing — ensuring the model is evaluated on completely unseen writers
- **Visualization**: t-SNE embedding visualization and image-mapped scatter plots for qualitative analysis of learned representations

## Dataset

**CEDAR Signature Dataset**
- 55 unique signers
- 24 genuine signatures per signer
- 24 skilled forgeries per signer
- Images resized to **113×80** pixels (grayscale, single channel)
- Input shape: `(80, 113, 1)`

## Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV (`cv2`)
- Pillow (`PIL`)
- scikit-learn
- albumentations
- matplotlib
- tqdm
- scipy

## Environment

The notebook is designed to run on **Google Colab** with GPU acceleration. The dataset is loaded from Google Drive.

## Usage

1. Upload the CEDAR dataset (`CEDAR.zip`) to your Google Drive under `Dataset/`
2. Open `Signature.ipynb` in Google Colab
3. Enable GPU runtime (Runtime → Change runtime type → GPU)
4. Run all cells sequentially

The notebook will:
1. Mount Google Drive and extract the dataset
2. Preprocess all signature images (background removal, enhancement, grayscale conversion)
3. Split data into train (45 signers) and test (10 signers) sets
4. Build and train the Siamese network with contrastive distance loss
5. Evaluate the model and visualize the learned embedding space using t-SNE

## Project Structure

```
bsc-project/
├── Signature.ipynb   # Main notebook with full pipeline
└── README.md         # This file
```

## Evaluation

The model is evaluated using:
- **Adjusted Rand Score (ARS)**
- **Normalized Mutual Information (NMI)**
- **K-Means clustering** on learned embeddings
- **t-SNE visualization** of the embedding space
- **Confusion matrix** analysis

## License

This project is part of a BSc thesis. Please contact the author for usage permissions.
