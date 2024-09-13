
# Ecommerce Propensity ML Model

## Overview

This repository contains an end-to-end machine learning pipeline for building an Ecommerce Propensity Model. The goal of this model is to predict the likelihood of a customer making a purchase based on various behavioral and transactional features. This helps in personalizing marketing strategies and improving customer conversion rates.

## Table of Contents
- [Dataset](#dataset)
- [Modeling Pipeline](#modeling-pipeline)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Key Performance Indicators](#key-performance-indicators)
- [Future Enhancements](#future-enhancements)

## Dataset

The datasets used for training the model are as follows:
- **Raw Data**: Contains customer interaction data, such as behavioral and transactional features. You can access the dataset [here](https://www.kaggle.com/datasets/adithiav/e-commerce-customer-behavior-data/data).
- **RFM Feature Dataset**: Includes Recency, Frequency, and Monetary (RFM) features alongside the raw data. You can access the RFM-enhanced dataset [here](https://www.kaggle.com/datasets/adithiav/customer-data-with-rfm-feature).

## Modeling Pipeline

1. **Exploratory Data Analysis (EDA)**:
   - Initial data exploration to understand the data distribution, patterns, and relationships.
   - Visualization of key features such as purchase frequency and customer behavior using **Matplotlib** and **Seaborn**.
   - Identification of missing values, outliers, and any data quality issues.
   - Correlation analysis to identify relationships between variables.

2. **Data Preprocessing**:
   - Handling missing values, outlier detection, and scaling numeric features.
   - Encoding categorical variables (e.g., one-hot encoding).
   - Feature selection based on correlation, domain knowledge, and importance ranking.

3. **RFM Feature Engineering**:
   - **Recency (R)**: Calculated as the number of days since the customer’s last transaction.
   - **Frequency (F)**: The total number of transactions made by the customer within a defined period.
   - **Monetary (M)**: The total amount of money spent by the customer.
   - These RFM features were added to the dataset as part of the feature engineering process to enhance the model’s predictive power.

4. **Model Selection**:
   - Various machine learning algorithms were experimented with, including:
     - Logistic Regression
     - Random Forest
     - XGBoost
     - LightGBM
   - Cross-validation was used to fine-tune hyperparameters.

5. **Evaluation**:
   - The models were evaluated using metrics such as Accuracy, AUC-ROC, Precision, Recall, and F1 Score.
   - The best-performing model was selected for deployment.

6. **Prediction and Inference**:
   - The trained model can be used to predict the likelihood of a customer making a purchase based on input features.

## Technologies Used

- **Python** (for scripting and ML modeling)
- **Pandas** and **NumPy** (for data manipulation and processing)
- **Scikit-learn**, **XGBoost**, **LightGBM** (for machine learning)
- **Matplotlib** and **Seaborn** (for data visualization)
- **Jupyter Notebook** (for exploratory analysis)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/adithia73/Ecommerce-Propensity-ML-Model.git
   cd Ecommerce-Propensity-ML-Model
   ```

## Key Performance Indicators

The key performance indicators for the model include:
- **Accuracy**: How well the model predicts purchase behavior.
- **AUC-ROC**: Measures the performance of the classification model at different threshold settings.
- **F1 Score**: The harmonic mean of precision and recall.

## Future Enhancements

- Implement real-time prediction APIs for integration with an eCommerce platform.
- Conduct A/B testing to assess model impact on business metrics like conversion rate and revenue.
