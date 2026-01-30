# ANN Churn Prediction

## Overview

This project provides a customer churn prediction system powered by an Artificial Neural Network (ANN). It includes a Streamlit web app for interactive predictions, trained model artifacts, and notebooks used for experimentation and inference.

## Features

- Predict churn probability based on customer profile inputs.
- Interactive Streamlit UI for real-time inference.
- Preprocessing pipeline using saved encoders and scaler.
- Reproducible experiments in notebooks.

## Project Structure

- [app.py](app.py) — Streamlit application for churn prediction.
- [Model.h5](Model.h5) — Trained ANN model.
- [requirements.txt](requirements.txt) — Python dependencies.
- [Notebook/experiments.ipynb](Notebook/experiments.ipynb) — Data exploration and model training workflow.
- [Notebook/Prediction.ipynb](Notebook/Prediction.ipynb) — Example inference workflow.
- [Notebook/Churn_Modelling.csv](Notebook/Churn_Modelling.csv) — Dataset used for training and experiments.

## Requirements

- Python 3.9 or later recommended
- Dependencies listed in [requirements.txt](requirements.txt)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

## Running the App

Start the Streamlit application from the project root:

streamlit run app.py

The app will load the model and preprocessing artifacts and present a form to generate churn predictions.

## Model Artifacts

The app expects the following files in the project root:

- [Model.h5](Model.h5)
- [Label_Encoder_Gender.pkl](Label_Encoder_Gender.pkl)
- [OHE_Geography.pkl](OHE_Geography.pkl)
- [Scaler.pkl](Scaler.pkl)

If these files are missing, generate them by running the training workflow in [Notebook/experiments.ipynb](Notebook/experiments.ipynb).

## Notebooks

- Use [Notebook/experiments.ipynb](Notebook/experiments.ipynb) to reproduce preprocessing and training steps.
- Use [Notebook/Prediction.ipynb](Notebook/Prediction.ipynb) to verify inference outside the app.

## Input Features

The prediction pipeline uses customer attributes such as `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, and `EstimatedSalary`.
