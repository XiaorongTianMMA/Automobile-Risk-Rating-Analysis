**Vehicle Risk Rating Analysis**

Overview

This repository contains the code and documentation for my individual project on vehicle risk rating analysis. The project involves building a Classification Model and a Clustering Model, with an emphasis on understanding and interpreting business implications.

**Project Components**

**Part 1: Classification Model**

Data Preprocessing: Removal of rows with missing values, filtering dataset entries, exclusion of irrelevant data, and conversion of categorical variables into dummy variables.
Feature Selection: Utilized lasso regression for significant predictor identification.
Model Building: Implementation of a Random Forest Classifier, hyperparameter tuning using GridSearch CV, and ROC curve analysis for threshold optimization.

**Part 2: Clustering Model**

Data Preprocessing: Similar approach to the Classification Model, with the addition of Min-Max Scaling.
PCA Analysis: Performed to enhance the performance of K-means clustering.
Model Building: Application of the elbow method for optimal k selection, and transition to k-prototype clustering due to the presence of multiple categorical variables.

**Part 3: Business Interpretation**

Analysis of key factors influencing the outcomes of the models and their potential applications in automated decision-making for stakeholders.
