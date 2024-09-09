# Fruits and Vegetables Identification Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Demo](#demo)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Tech Stack](#tech-stack)
7. [Results](#results)

## Project Overview

This project aims to identify different types of fruits and vegetables using image classification techniques. The model classifies over 30 categories, including apples, bananas, cabbages, tomatoes, and more.

## Features

- Real-time classification of fruits and vegetables.
- Easy-to-use web interface for uploading images.
- High accuracy and fast predictions.

## Demo

You can try out the model and see its predictions in action at the following link: [Streamlit App](https://imageclassifierml.streamlit.app/)

## Dataset

The dataset contains thousands of images labeled by fruit and vegetable types. It was obtained from Kaggle: [Fruit and Vegetable Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition). This dataset was used to train the convolutional neural network.

## Model Architecture

The model used for this project is a Convolutional Neural Network (CNN) designed to classify fruits and vegetables. Below is the detailed architecture:

- **Rescaling Layer**: This layer normalizes pixel values by scaling them between 0 and 1.
- **Conv2D Layers**: Three convolutional layers with 64, 128, and 256 filters, each with a 3x3 kernel size, `same` padding, and ReLU activation for feature extraction.
- **MaxPooling2D Layers**: Used after each Conv2D layer to downsample the feature maps and reduce computational complexity.
- **Flatten Layer**: Flattens the multi-dimensional feature maps into a 1D array to be used by fully connected layers.
- **Dropout Layer**: A dropout rate of 30% is applied to prevent overfitting during training.
- **Dense Layers**: 
  - A fully connected dense layer with 256 units and ReLU activation.
  - The output layer has neurons equal to the number of categories for classification.

The model is compiled with:
- **Optimizer**: Adam optimizer for efficient training.
- **Loss Function**: Sparse Categorical Crossentropy, since this is a multi-class classification problem.
- **Metrics**: Accuracy, used to measure the model’s performance.

## Tech Stack

The following technologies were used in building and deploying this project:

- **TensorFlow / Keras**: For building and training the CNN model.
- **NumPy**: For data preprocessing and handling.
- **PIL (Python Imaging Library)**: For image processing tasks.
- **Streamlit**: To create a user-friendly web interface for the model.
- **Matplotlib**: For visualizing model performance during training.

## Results

The model achieved an accuracy of **97%** on the test dataset, demonstrating its effectiveness for real-world applications like inventory management or dietary apps.
