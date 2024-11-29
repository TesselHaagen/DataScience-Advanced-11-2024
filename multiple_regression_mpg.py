"""
multiple_regression_mpg.py

Works with an external repository to retrieve the MPG data.
"""
# %% Step 0: Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# %% Step 1: Get the MPG dataset
raw_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                          delim_whitespace=True,
                          names=["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration",
                                 "model_year", "origin", "car_name"])

#%% Step 2: Exploratory Data Analysis (EDA)
print(raw_df.head())  # Display first few rows
print(raw_df.describe())  # Summary statistics

# Remove the entries where the horsepower is marked with a question mark
auto_mpg_df = raw_df[raw_df.horsepower != '?']

#%% Step 3: Encode categorical features (if any)
# Assuming "origin" is categorical and needs encoding
auto_mpg_df = pd.get_dummies(auto_mpg_df, columns=["origin"], drop_first=True)

# Remove the horsepower columns without useful values
auto_mpg_df = auto_mpg_df[auto_mpg_df.horse_power != '?']

#%% Step 4: Split into Training and Test dataset
X = auto_mpg_df.drop(columns=["mpg", "car_name"])  # Features
y = auto_mpg_df["mpg"]  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Step 5: Perform Linear Regression with mpg as target
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)
linear_reg_mse = mean_squared_error(y_test, y_pred)
print("Linear Regression MSE:", linear_reg_mse)

#%% Step 6: Scale the numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Step 7: Perform Linear Regression on the scaled dataset
linear_reg_scaled = LinearRegression()
linear_reg_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = linear_reg_scaled.predict(X_test_scaled)
linear_reg_scaled_mse = mean_squared_error(y_test, y_pred_scaled)
print("Linear Regression on scaled data MSE:", linear_reg_scaled_mse)

#%% Step 8: Perform Lasso Regression on the scaled dataset
lasso_reg = Lasso(alpha=0.1)  # Set alpha (regularization strength)
lasso_reg.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_reg.predict(X_test_scaled)
lasso_reg_mse = mean_squared_error(y_test, y_pred_lasso)
print("Lasso Regression MSE:", lasso_reg_mse)

#%% Step 9: Perform Random Forest Regression on the dataset
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
rf_reg_mse = mean_squared_error(y_test, y_pred_rf)
print("Random Forest Regression MSE:", rf_reg_mse)

#%% Step 10: Determine the most important features
importances = rf_reg.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
print("Most important features:")
print(feature_importance_df.head())

#%% Step 11: Determine the least important features
print("Least important features:")
print(feature_importance_df.tail())

#%% Step 12: Perform Linear Regression with fewer features
# Let's use the top 5 most important features
selected_features = feature_importance_df.head()["Feature"].tolist()
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

linear_reg_selected = LinearRegression()
linear_reg_selected.fit(X_train_selected, y_train)
y_pred_selected = linear_reg_selected.predict(X_test_selected)
linear_reg_selected_mse = mean_squared_error(y_test, y_pred_selected)
print("Linear Regression with selected features MSE:", linear_reg_selected_mse)

#%% Step 13: Interpret the results
# This step involves analyzing the coefficients of the linear models, comparing MSE values, and understanding feature importances.
