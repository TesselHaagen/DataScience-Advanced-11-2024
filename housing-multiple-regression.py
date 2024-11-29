# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:18:35 2024

@author: linac
"""

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Variable to predict
target_column: str = "median_house_value"

# %% 1. Import data
df: pd.DataFrame = pd.DataFrame(load_dataset("leostelon/california-housing")['train'])

# 2. Explore the data
# print(df.head())
# print(df[['housing_median_age', 'total_rooms', 'total_bedrooms']].describe())
print(df.columns)

# Plotting
sns.pairplot(df)
plt.show()

# 3. Clean the data. Remove missing values. (really depends on the dataset, this is a simple implementation)
original_nonempty: pd.DataFrame = df.dropna()

# 4. Prepare the data for modelling.
print(f"Unieke waarden in ocean proximity {original_nonempty['ocean_proximity'].unique()}")

# Encode categorical variables
ocean_dummies = pd.get_dummies(original_nonempty['ocean_proximity'], prefix='ocean_')
df = pd.concat([original_nonempty, ocean_dummies], axis=1)
df = df.drop(columns="ocean_proximity")
print(f"Columns after dummy variable: {df.columns}")

# Scale numerical variables
scaler = StandardScaler()

# Extract numerical columns
numerical_columns: list = df.select_dtypes(include=['int', 'float']).columns

# Fit scaler to the numerical data
scaler.fit(df[numerical_columns])

# Transform and scale the numerical columns
df_scaled = df.copy()  # Create a copy of the original DataFrame
df_scaled[numerical_columns] = scaler.transform(df[numerical_columns])

# Determine target variable. Split in training and test sets.
X = df_scaled.drop(columns=[target_column])     # Features
y = df_scaled[target_column]                    # Target

# Make the train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def train_model(model, X_train, X_test, y_train, y_test) -> list:
    """
    Train a model on a given (prepared) test set.
    :param model:       Any model from sklearn extending the base estimator
    :param X:           Feature matrix
    :param y:           Target variable column
    :return:            List of predictions
    """
    # Create and train the model with default settings
    m = model()
    m.fit(X_train, y_train)

    y_pred = m.predict(X_test)

    mse: float = mean_squared_error(y_test, y_pred)
    print(f"{model} results:")
    print(f"Mean Squared Error na Logistische Regressie: {mse}")
    print(f"Model Coefficients: zip({m.coef_})")
    print()
    return y_pred


# Use regular Linear Regression
res = train_model(LinearRegression, X_train, X_test, y_train, y_test)
# Store the original results with the predicted results.
#s = pd.Series(res, index=y_test)

# Repeat the modeling with other models like Ridge, Lasso, ElasticNet.
# Also build a model with RandomForestRegressor and display the feature importances.
train_model(Lasso, X_train, X_test, y_train, y_test)
train_model(RandomForestRegressor, X_train, X_test, y_train, y_test)
