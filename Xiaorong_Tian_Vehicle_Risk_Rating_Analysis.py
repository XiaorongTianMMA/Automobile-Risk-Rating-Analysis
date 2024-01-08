# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:25:04 2023

@author: Xiaorong Tian
"""

# Xiaorong Tian Individual Project Part 1: Classification
### You can find the grading code at the end of this section as well as the accuracy ###
import pandas as pd
import matplotlib.pyplot as plt
# importing the libraries 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

kickstarter = pd.read_excel("C:/Users/95675/OneDrive/桌面/Data Mining and Visualization/Individual Project/Kickstarter.xlsx")

# Part 1: Data Pre-processing
kickstarter = kickstarter.dropna()
kickstarter = kickstarter[kickstarter['state'].isin(['successful', 'failed'])]

kickstarter['goal_usd'] = kickstarter['goal'] / kickstarter['static_usd_rate']

kickstarter = kickstarter.drop(columns = ['goal','id', 'name', 'pledged', 'backers_count', 
'static_usd_rate', 'usd_pledged', 'state_changed_at', 'state_changed_at_weekday', 
'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 
'state_changed_at_hr', 'launch_to_state_change_days', 
'deadline', 'created_at', 'launched_at', 'static_usd_rate', 'spotlight'])


categorical_columns = ['category', 'disable_communication', 'country', 'currency', 
'staff_pick', 'deadline_weekday',  'created_at_weekday', 'launched_at_weekday']
kickstarter_dummies = pd.get_dummies(kickstarter[categorical_columns]).astype(int)

kickstarter_full = pd.concat([kickstarter, kickstarter_dummies], axis=1)

kickstarter = kickstarter_full.drop(columns=categorical_columns)

kickstarter['state'] = kickstarter['state'].map({'failed': 0, 'successful': 1})

# Part 2: Feature Selection for Classification 

y = kickstarter['state']
x = kickstarter.drop(columns = ['state'])
# Lasso
best_score = 10000
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

alphas = np.linspace(1e-5, 1, 100)

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    if mse < best_score:
        best_score = mse
        best_alpha = alpha

print("Best alpha:", best_alpha)

lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(x_train, y_train)

selected_features = [feature for feature, coef in zip(x.columns, lasso_best.coef_) if coef != 0]
print("Selected features:", selected_features)
kickstarterC = kickstarter[selected_features]
kickstarterC['state'] = kickstarter['state']

y = kickstarterC['state']
x = kickstarterC.drop(columns = ['state'])

# Part 3: Classification Model
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

# Initial RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model_rf = randomforest.fit(x_train, y_train)

# Make prediction and evaluate accuracy
y_test_pred = model_rf.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_test_pred)
print(f"Initial Model Accuracy: {accuracy_rf}")

# Grid Search for Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 5, 10, 20],      
    'min_samples_split': [2, 5, 10],      
    'min_samples_leaf': [1, 2, 4],        
}

grid_search = GridSearchCV(estimator=randomforest, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Best model from grid search
best_model = grid_search.best_estimator_



accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_test_pred)
print("Recall:", recall)
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_test_pred)
print("Precision:", precision)
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_test_pred)
print("F1 Score:", f1)

# AUC Graph
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
best_model = grid_search.best_estimator_

# Get predicted probabilities
y_test_prob = best_model.predict_proba(x_test)[:, 1]

# Generate ROC curve values: fpr, tpr, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

# Calculate Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Find the optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Best Threshold:", optimal_threshold)
# New threshold:
y_test_prob = best_model.predict_proba(x_test)[:, 1]

# Define a new threshold
new_threshold = 0.33

# Apply the new threshold to make new predictions
y_test_pred_new_threshold = (y_test_prob >= new_threshold).astype(int)

# Calculate the recall, precision, and F1 score using this new set of predictions
recall_new = recall_score(y_test, y_test_pred_new_threshold)
precision_new = precision_score(y_test, y_test_pred_new_threshold)
f1_new = f1_score(y_test, y_test_pred_new_threshold)
accuracy_new = accuracy_score(y_test, y_test_pred_new_threshold)
print("Accuracy with New Threshold:", accuracy_new)
print("Recall with New Threshold:", recall_new)
print("Precision with New Threshold:", precision_new)
print("F1 Score with New Threshold:", f1_new)

# Assuming best_model is the final model after GridSearchCV
best_model = grid_search.best_estimator_

# Get feature importances
importances = best_model.feature_importances_

# Get feature names
feature_names = x.columns

# Combine into a DataFrame
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort by importance and select top 10
top_features = feature_importances.sort_values(by='importance', ascending=False).head(10)

print(top_features)


### Grading Part ###

import pandas as pd
import matplotlib.pyplot as plt
# importing the libraries 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Change the file path here to run the grading dataset
kickstarterG = pd.read_excel("C:/Users/95675/OneDrive/桌面/Data Mining and Visualization/Individual Project/Kickstarter-Grading.xlsx")

kickstarterG = kickstarterG.dropna()
kickstarterG = kickstarterG[kickstarterG['state'].isin(['successful', 'failed'])]
kickstarterG['goal_usd'] = kickstarterG['goal'] / kickstarterG['static_usd_rate']
kickstarterG = kickstarterG.drop(columns = ['goal','id', 'name', 'pledged', 'backers_count', 
'static_usd_rate', 'usd_pledged', 'state_changed_at', 'state_changed_at_weekday', 
'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 
'state_changed_at_hr', 'launch_to_state_change_days', 
'deadline', 'created_at', 'launched_at', 'static_usd_rate', 'spotlight'])


categorical_columns = ['category', 'disable_communication', 'country', 'currency', 
'staff_pick', 'deadline_weekday',  'created_at_weekday', 'launched_at_weekday']
kickstarterG_dummies = pd.get_dummies(kickstarterG[categorical_columns]).astype(int)

kickstarterG_full = pd.concat([kickstarterG, kickstarterG_dummies], axis=1)

kickstarterG = kickstarterG_full.drop(columns=categorical_columns)

kickstarterG['state'] = kickstarterG['state'].map({'failed': 0, 'successful': 1})

# These features are selected based on the LASSO feature selection I used in the previous code
selected_features = ['state',
    'name_len', 'name_len_clean', 'blurb_len', 'blurb_len_clean', 
    'deadline_month', 'deadline_day', 'deadline_yr', 'deadline_hr', 
    'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr', 
    'launched_at_month', 'launched_at_day', 'launched_at_yr', 'launched_at_hr', 
    'create_to_launch_days', 'launch_to_deadline_days', 'goal_usd', 
    'category_Academic', 'category_Apps', 'category_Blues', 
    'category_Experimental', 'category_Festivals', 'category_Flight', 
    'category_Gadgets', 'category_Hardware', 'category_Immersive', 
    'category_Makerspaces', 'category_Musical', 'category_Places', 
    'category_Plays', 'category_Robots', 'category_Shorts', 
    'category_Software', 'category_Sound', 'category_Spaces', 
    'category_Thrillers', 'category_Wearables', 'category_Web', 
    'category_Webseries', 'country_AT', 'country_AU', 'country_BE', 
    'country_DE', 'country_DK', 'country_ES', 'country_FR', 'country_GB', 
    'country_IE', 'country_IT', 'country_LU', 'country_NL', 'country_NO',
    'country_NZ', 'country_SE', 'country_US', 'currency_AUD', 'currency_DKK', 
    'currency_EUR', 'currency_GBP', 'currency_NOK', 'currency_NZD', 
    'currency_SEK', 'currency_USD', 'deadline_weekday_Monday', 
    'deadline_weekday_Saturday', 'deadline_weekday_Sunday', 
    'deadline_weekday_Thursday', 'deadline_weekday_Tuesday', 
    'deadline_weekday_Wednesday', 'created_at_weekday_Friday', 
    'created_at_weekday_Monday', 'created_at_weekday_Saturday', 
    'created_at_weekday_Thursday', 'created_at_weekday_Tuesday', 
    'created_at_weekday_Wednesday', 'launched_at_weekday_Friday', 
    'launched_at_weekday_Monday', 'launched_at_weekday_Saturday', 
    'launched_at_weekday_Sunday', 'launched_at_weekday_Thursday', 
    'launched_at_weekday_Tuesday'
]
for column in selected_features:
    if column not in kickstarterG.columns:
        kickstarterG[column] = 0

# Reorder columns in kickstarterG to match the order in selected_features
kickstarterG = kickstarterG.reindex(columns=selected_features)
X_grading = kickstarterG.drop(columns=['state'])
y_grading = kickstarterG['state']

# Using the best_model to make predictions on the new dataset
y_grading_pred = best_model.predict(X_grading)

# Calculating accuracy
from sklearn.metrics import accuracy_score
grading_accuracy = accuracy_score(y_grading, y_grading_pred)
print("Accuracy on Grading Dataset:", grading_accuracy)


# Accuracy on Grading Dataset: 0.7647798742138365


# Xiaorong Tian Individual Project Part 2: Clustering
import pandas as pd
import matplotlib.pyplot as plt
# importing the libraries 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

kickstarter = pd.read_excel("C:/Users/95675/OneDrive/桌面/Data Mining and Visualization/Individual Project/Kickstarter.xlsx")

# Part 1: Data Pre-processing
kickstarter = kickstarter.dropna()
kickstarter['goal_usd'] = kickstarter['goal'] / kickstarter['static_usd_rate']
kickstarter = kickstarter[kickstarter['state'].isin(['successful', 'failed'])]
kickstarter = kickstarter.drop(columns = ['goal','id', 'name', 'pledged', 'backers_count', 
'static_usd_rate', 'usd_pledged', 'state_changed_at', 'state_changed_at_weekday', 
'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 
'state_changed_at_hr', 'launch_to_state_change_days', 
'deadline', 'created_at', 'launched_at', 'static_usd_rate', 'spotlight'])


categorical_columns = ['category', 'disable_communication', 'country', 'currency', 
'staff_pick', 'deadline_weekday',  'created_at_weekday', 'launched_at_weekday']


# Standardizing numerical columns using Min-Max Scaler
from sklearn.preprocessing import MinMaxScaler
numerical_columns = kickstarter.select_dtypes(include=['int64', 'float64']).columns
scaler = MinMaxScaler()
kickstarter[numerical_columns] = scaler.fit_transform(kickstarter[numerical_columns])

kickstarter['state'] = kickstarter['state'].map({'failed': 0, 'successful': 1})

# Part 2 PCA Analysis for Dimension Reduction
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Applying PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
kickstarter_pca = pca.fit_transform(kickstarter[numerical_columns])

# Examining the Explained Variance
explained_variance = pca.explained_variance_ratio_
print("Explained Variance per Principal Component: ", explained_variance)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title("Explained Variance by Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.show()

# Part 3: K-Means Clustering
from sklearn.cluster import KMeans
import numpy as np

WCSS = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(kickstarter_pca)
    WCSS.append(kmeans.inertia_)

# Plot the WCSS to observe the elbow
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), WCSS, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Choose the number of clusters (k) based on the Elbow Method
k = 3

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(kickstarter_pca)

kickstarter['Cluster_Labels'] = cluster_labels

print(kickstarter['Cluster_Labels'].value_counts())

# Plotting the cluster
plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(kickstarter_pca[cluster_labels == i, 0], kickstarter_pca[cluster_labels == i, 1], 
                label=f'Cluster {i+1}', c=colors[i], marker='o', edgecolor='black', s=50)

# Plot the centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='*', c='yellow', edgecolor='black', label='Centroids')

plt.title('Clusters of Kickstarter Projects')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Part 4: Silhouette Method
from sklearn.metrics import silhouette_score

silhouette_scores = [] 
# Range of k we want to try
range_n_clusters = list(range(2, 11))

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    cluster_labels = clusterer.fit_predict(kickstarter_pca)
    
    # The silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(kickstarter_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the silhouette score is : {silhouette_avg}")

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Part 5 : K-prototype clustering

from kmodes.kprototypes import KPrototypes

categorical_indices = [kickstarter.columns.get_loc(col) for col in categorical_columns]

X = kickstarter.values

# Initialize and fit the model
kmixed = KPrototypes(n_clusters=3)
clusters = kmixed.fit_predict(X, categorical=categorical_indices)
print(kickstarter['Cluster_Labels'].value_counts())


# Part 6: Insights Making
# Cluster Characteristics
numerical_features = ['goal_usd', 'name_len', 'name_len_clean', 'blurb_len', 
'blurb_len_clean','deadline_month', 'deadline_day', 'deadline_yr', 'deadline_hr',
'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr',
'launched_at_month', 'launched_at_day', 'launched_at_yr', 'launched_at_hr',
'create_to_launch_days', 'launch_to_deadline_days']

# Iterate through each feature and print the average for each cluster
for feature in numerical_features:
    print(f"\nAverage values for {feature}:")
    for i in range(kmixed.n_clusters):
        avg_value = kickstarter[kickstarter['Cluster_Labels'] == i][feature].mean()
        print(f"Cluster {i}: {avg_value}")
        
# List of categorical features
categorical_columns = ['state', 'category', 'disable_communication', 'country', 'currency', 
'staff_pick', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday']

# Iterate through each categorical feature and print the crosstab with cluster labels
for feature in categorical_columns:
    print(f"\nCrosstab for {feature}:")
    cross_tab = pd.crosstab(kickstarter[feature], kickstarter['Cluster_Labels'])
    print(cross_tab)



