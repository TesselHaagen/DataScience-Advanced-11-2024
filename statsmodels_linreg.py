"""
statsmodels_linreg.py

This script shows off the statsmodel API in combination with a linear regression.
While this module is not the main focus of this course, it has advantages for
analytical tasks.

The statsmodel API is particularly interesting due to the detailed information it provides
about its models and their performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Pearson's father and son dataset (can be done with regular Pandas DataFrame)
data = np.genfromtxt('../data/pearson-father-son.csv', delimiter=',', skip_header=1)

# Extract father's height (feature) and son's height (target)
X = data[:, 1].reshape(-1, 1)  # Reshape to a column vector
y = data[:, 2]

# Add a constant term for the intercept
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the OLS model
model = sm.OLS(y_train, X_train)
results = model.fit()

# Make predictions on the testing data
y_pred = results.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot the data points and the regression line
plt.scatter(X_test[:, 1], y_test, color='blue', label='Actual data')
plt.plot(X_test[:, 1], y_pred, color='red', linewidth=2, label='Linear regression')
plt.xlabel("Father's Height")
plt.ylabel("Son's Height")
plt.title("Linear Regression: Father's Height vs Son's Height")
plt.legend()
plt.show()

# Print the summary of the regression results
print(results.summary())
