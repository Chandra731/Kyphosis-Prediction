# Kyphosis Prediction with Decision Tree

This project uses a decision tree model to predict the presence (or absence) of Kyphosis, a spinal deformity in children. It involves exploring the data, building and evaluating a decision tree classifier, and visualizing the decision-making process.

## Description

Kyphosis is a spinal condition characterized by an abnormal forward curvature of the spine. Early detection and treatment are crucial for managing the condition effectively. This project aims to develop a machine learning model that can help predict whether a child will develop Kyphosis based on various medical features.

## Data Source

The dataset used for this project is the "Kyphosis" dataset, which is publicly available on Kaggle: [Kyphosis Dataset](https://www.kaggle.com/datasets/abbasit/kyphosis-dataset)

## Features

The dataset includes the following features:

- `Age`: Age of the patient in months
- `Number`: Number of vertebrae involved
- `Start`: Starting vertebrae of the Kyphosis
- `Kyphosis`: Whether Kyphosis is present (1) or absent (0)

## Usage

1. **Clone the Repository:**
   git clone https://github.com/brucelee31072004/Kyphosis-Prediction.git
2. **Install Dependencies:**
   pip install pandas seaborn matplotlib scikit-learn pydot
3. **Run the Code:**
   python kyphosis_prediction.py

## Results
The decision tree model achieves an accuracy of approximately 84% on the test data. 
**Classification Report**
             precision    recall  f1-score   support

      absent       0.84      0.94      0.89        17
     present       0.83      0.62      0.71         8

    accuracy                           0.84        25
   macro avg       0.84      0.78      0.80        25
weighted avg       0.84      0.84      0.83        25

** Confusion Matrix**
   [[16  1]
   [ 3  5]]
