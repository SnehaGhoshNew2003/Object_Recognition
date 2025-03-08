# CIFAR-10 Object Recognition using ResNet50

## Overview
This project classifies images from the **CIFAR-10 dataset** using **ResNet50**, a powerful deep learning model for image recognition. The dataset consists of 60,000 images categorized into 10 different classes.

## Dataset
- **Source**: [CIFAR-10 Dataset](https://www.kaggle.com/competitions/cifar-10/overview)
- **Classes**:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

## Dependencies
To run this project, install the required dependencies:
```bash
pip install tensorflow keras numpy pandas matplotlib kaggle
```

## Setup & Execution
1. Clone or download the repository.
2. Download the dataset from Kaggle:
   ```bash
   kaggle competitions download -c cifar-10
   ```
3. Extract the dataset:
   ```bash
   unzip cifar-10.zip -d cifar-10
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook CIFAR_10_Object_Recognition_using_ResNet50_.ipynb
   ```
5. Follow the notebook cells to preprocess data, train the model, and evaluate performance.

## Model Architecture
- **Preprocessing**:
  - Image resizing and normalization.
  - Splitting into training and testing sets.
- **ResNet50 Model**:
  - Pretrained on **ImageNet**.
  - Fine-tuned for CIFAR-10 classification.
  - Uses **Softmax activation** for multi-class classification.
- **Loss Function**: Categorical Crossentropy.
- **Optimizer**: Adam.

## Results
- The model achieves high accuracy in classifying CIFAR-10 images.
- Performance is evaluated using accuracy and loss curves.

## Future Improvements
- Experiment with different architectures like **EfficientNet**.
- Use **data augmentation** to improve generalization.
- Deploy the model using **Flask** or **FastAPI** for real-time classification.
