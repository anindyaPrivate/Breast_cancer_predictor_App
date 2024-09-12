# Breast Cancer Predictor 🩺
This is a web-based application that predicts the likelihood of breast cancer based on medical features. The app uses a machine learning model to classify tumors as either Malignant (Cancerous) or Benign (Not Cancerous).

## Table of Contents
Demo
Features
Installation
Usage
Technologies Used
Screenshots

## Features
 Predict Tumor Type: Based on input features like radius, texture, perimeter, etc., the app predicts whether a tumor is cancerous or not.
 Interactive UI: User-friendly interface created using Bootstrap 5, which includes cards, images, and responsive design.
 Image Output: Displays a relevant image for both cancerous and non-cancerous predictions.

## Installation
To run this application locally, follow these steps:

Clone the repository:
```bash
git clone https://github.com/anindyaPrivate/Breast_cancer_predictor_App.git
```
Navigate to the project directory:
```bash
cd Breast_cancer_predictor_App
```
Run the application:
```bash
flask run
```
### Open your browser and go to http://localhost:5000.

## Usage
Input medical breast cancer features like radius, texture, perimeter, etc.

Click on the Predict button to get a prediction.

View the result, which will display whether the tumor is Malignant (Cancerous) or Benign (Not Cancerous) along with an image and description.

## Technologies Used
Flask: Backend web framework to handle requests and serve the model.

Bootstrap 5: Frontend framework for responsive design and styling.

Machine Learning: Trained model for breast cancer prediction.

HTML5/CSS3: For creating the structure and style of the web pages.

## Screenshots
Input Page:
