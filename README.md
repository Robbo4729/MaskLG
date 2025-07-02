# MaskLG Project

## Overview
MaskLG is a deep learning framework designed for extracting and utilizing Learngene parameters from a pre-trained ancestor model. The framework implements a Vision Transformer architecture with modifications for masked training, allowing for efficient parameter selection and model distillation.

## Features
- **Vision Transformer Architecture**: Implements the VisionTransformer class with support for masked training.
- **Masked Blocks**: Utilizes Gumbel-Softmax to create binary masks for model parameters, enabling selective parameter retention.
- **Distillation Support**: Includes distilled versions of the Vision Transformer for improved performance.
- **Custom Data Loaders**: Implements RASampler for distributed data loading with repeated augmentations.
- **Loss Functions**: Combines standard loss functions with knowledge distillation loss for enhanced training.
- **Training and Evaluation**: Provides functions for training the model and evaluating its performance.

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd MaskLG
pip install -r requirements.txt
```

## Usage
To run the main application, execute the following command:

```bash
python src/main.py --model deit_base_patch16_224 --epochs 10 --lr 0.01
```

### Command Line Arguments
- `--model`: Specify the model architecture to use (e.g., `deit_base_patch16_224`).
- `--epochs`: Number of training epochs.
- `--lr`: Learning rate for the optimizer.
- `--temperature`: Temperature parameter for Gumbel-Softmax.
- `--threshold`: Threshold for binary mask generation.

## File Descriptions
- **src/__init__.py**: Marks the directory as a Python package.
- **src/main.py**: Entry point for the application, sets up the model and extracts Learngene parameters.
- **src/models.py**: Defines the VisionTransformer class and related architectures.
- **src/MaskLG_vit.py**: Contains the VisionTransformer implementation with masked training modifications.
- **src/masked_blocks.py**: Implements the MaskedBlock class for parameter masking.
- **src/des_models.py**: Defines distilled versions of the VisionTransformer.
- **src/datasets.py**: Functions to build datasets and transformations.
- **src/samplers.py**: Implements the RASampler for distributed data loading.
- **src/losses.py**: Implements the DistillationLoss class for training.
- **src/engine.py**: Contains training and evaluation functions.
- **src/utils.py**: Provides utility functions and classes for metrics tracking.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.