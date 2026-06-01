# Satellite Image Classification using CNN and PyTorch

This project implements an end-to-end deep learning pipeline for classifying satellite images into 10 terrain categories using a custom Convolutional Neural Network (CNN) built with PyTorch.

### Live Demo
The Shiny web application can be accessed here: https://connect.posit.cloud/xhesikagjyla/content/019e8287-ca96-be2d-8d3c-b6adf51d6080

You can test the deployed application using your own satellite images.
For convenience, sample images are available in the repository inside the:

```text
public_test_data/
```

directory.

Simply download any image from this folder and upload it through the web interface. The application will:

* Display the uploaded image
* Predict the terrain class
* Show the confidence score of the prediction
* Display a probability table containing the predicted probabilities for all classes

This provides an easy way to explore the model's performance on unseen satellite imagery.

### Dataset
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


### Web Application

* Upload satellite images
* Real-time prediction
* Confidence score display
* Class probability table
* Image preview

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

