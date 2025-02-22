# Heart Disease Detection

This repository contains a machine learning project aimed at predicting the likelihood of heart disease in individuals based on various health-related features such as age, sex, cholesterol levels, and more. The goal of the project is to build an effective model that can assist in early detection of heart disease.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)


## Project Overview
This project aims to use machine learning algorithms to predict heart disease risk. The dataset includes features like age, sex, blood pressure, cholesterol, and other factors that may influence heart disease.

### Objective:
- Predict whether an individual has heart disease or not.
- Evaluate the performance of different machine learning models.
- Provide a user-friendly model for heart disease prediction.

## Dataset
The dataset used for training the model is the **Heart Disease UCI dataset**. It contains the following features:

- `age`: Age of the person
- `sex`: Sex of the person (1 = male, 0 = female)
- `cp`: Chest pain type (4 values)
- `trestbps`: Resting blood pressure
- `chol`: Serum cholesterol level
- `fbs`: Fasting blood sugar (1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1 = yes, 0 = no)
- `oldpeak`: Depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy
- `thal`: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
- `target`: Heart disease diagnosis (1 = yes, 0 = no)

The dataset can be found in this [Heart Disease UCI Dataset link](https://archive.ics.uci.edu/ml/datasets/heart+disease).

## Model Training
The following models are used to predict heart disease:

- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

## Model Evaluation
For each model, the following metrics are evaluated:

- **Accuracy**
- **Confusion Matrix**
- **Classification Report**
- **ROC Curve and AUC Score**

## Model
The machine learning models in this project are built using the Scikit-learn library. These models include:

- **Logistic Regression**
- **Random Forest Classifier**
- **SVM (Support Vector Machine)**
- **KNN (K-Nearest Neighbors)**

We evaluate each model's performance based on various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. The final model is selected based on the best performance (**Random Forest in this case**).

## Results
The model results, including accuracy, confusion matrix, and classification report, are as follows:

- **Accuracy**: 85%
- **Precision**: 83%
- **Recall**: 80%
- **F1-Score**: 81%

The **Random Forest Classifier** performed the best among all tested models.

### ROC Curve:
The ROC curve and AUC score are used to evaluate the performance of each model. The model with the highest AUC score is selected for deployment.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork this repository, create a pull request, or open an issue.

### To contribute:
1. **Fork** the repository.
2. **Create a new branch**.
3. **Make changes** and commit them.
4. **Push to your fork**.
5. **Create a pull request**.
