"""
linear_regression_comparison.py

This file takes the following steps:
1. Implement a linear model
2. Set some X and y values
3. Add a little noise to the y values
4. Use a polynomial fit with NumPy and SciPy
5. Compare the results through printing and plotting

What can we conclude from this experiment?
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define linear model function
def linear_model(x, m, c):
    return m * x + c


# Select range of x values
x_values = np.linspace(0, 10, 100)

# Determine corresponding y values using linear model
m_true = 2
c_true = 5
y_values_true = linear_model(x_values, m_true, c_true)

# Add noise to y values
noise = np.random.normal(0, 1, len(x_values))
y_values_noisy = y_values_true + noise

# Use numpy.polyfit to determine linear regression coefficients
coefficients_np = np.polyfit(x_values, y_values_noisy, 1)

# Use scipy.curve_fit to determine linear regression coefficients
popt, _ = curve_fit(linear_model, x_values, y_values_noisy)

# Compare the results
print("Numpy polyfit coefficients:", coefficients_np)
print("Scipy curve_fit coefficients:", popt)

# Plot the data and linear regression lines
plt.scatter(x_values, y_values_noisy, label='Noisy Data')
plt.plot(x_values, linear_model(x_values, *coefficients_np), color='red', label='Numpy Polyfit')
plt.plot(x_values, linear_model(x_values, *popt), color='green', label='Scipy Curve_fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Comparison')
plt.legend()
plt.show()
