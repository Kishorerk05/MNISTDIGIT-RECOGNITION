# MNIST Dataset

The MNIST (Modified National Institute of Standards and Technology) dataset is a large collection of handwritten digits commonly used for training various image processing systems.

## Dataset Overview

- **Classes**: 10 (digits 0–9)
- **Samples**: 70,000 images (60,000 for training, 10,000 for testing)
- **Image Size**: 28x28 pixels, grayscale
- **File Format**: PNG images or CSV files (flattened pixel values)

## Structure

The dataset consists of two parts:
1. **Training set**: 60,000 images and labels
2. **Test set**: 10,000 images and labels

Each image is a 28x28 matrix of pixel values ranging from 0 (black) to 255 (white).

## Usage

### Loading in Python

```python
from tensorflow.keras.datasets import mnist

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0
