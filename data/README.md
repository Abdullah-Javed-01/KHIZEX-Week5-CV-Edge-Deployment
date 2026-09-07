# Data

This project uses the public `sklearn.datasets.load_digits` handwritten-digit image-classification dataset. It contains 1,797 grayscale 8x8 images across 10 classes. The pipeline uses a fixed stratified 75/25 train/test split with random seed 42, so all model variants are evaluated on exactly the same held-out set.
