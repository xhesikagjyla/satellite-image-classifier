# Satellite Image Classification using CNN and PyTorch

This project implements an end-to-end deep learning pipeline for classifying satellite images into 10 terrain categories using a custom Convolutional Neural Network (CNN) built with PyTorch.

## Live Demo

The Shiny web application can be accessed here:

https://connect.posit.cloud/xhesikagjyla/content/019e8287-ca96-be2d-8d3c-b6adf51d6080

## Dataset

The model is trained on the EuroSAT RGB dataset consisting of 64×64 RGB satellite images belonging to the following classes:

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

## Project Features

### Data Processing

* Automatic dataset preprocessing
* Train/validation split with stratification
* Data augmentation and normalization

### Exploratory Data Analysis

* Random sample visualization
* RGB pixel value distributions
* Brightness analysis per class

### Deep Learning Model

* Custom CNN architecture built from scratch
* Batch Normalization
* Dropout regularization
* AdamW optimizer
* Learning rate scheduling
* Early stopping
* Best model checkpointing

### Evaluation

* Training and validation loss curves
* Validation accuracy tracking
* Confusion matrix
* Misclassified sample analysis

### Web Application

* Upload satellite images
* Real-time prediction
* Confidence score display
* Class probability table
* Image preview

## Results

The final model achieved approximately:

* Validation Accuracy: 95.4%
* Training Accuracy: 97.3%

The model demonstrates strong generalization performance while avoiding significant overfitting.

## Technologies Used

* Python
* PyTorch
* Torchvision
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Pillow
* Shiny for Python

## Repository Structure

```text
.
├── notebook/
│   └── k12359281.ipynb
│
├── app/
│   ├── app.py
│   └── assets/
│       └── weights/
│           └── best_model.pth
│
├── requirements.txt
└── README.md
```

## Running Locally

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the Shiny Application

```bash
shiny run --reload app/app.py
```

The application will open in your browser.

## Author

Edita Gjyla

MSc Artificial Intelligence

Johannes Kepler University Linz
