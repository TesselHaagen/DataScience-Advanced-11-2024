import numpy as np
import matplotlib.pyplot as plt

# Define the polynomial function
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

# Define the gradient function (derivative of the polynomial function)
def gradient_function(x, a, b):
    return 2 * a * x + b

# Define the derivative of the gradient function (second derivative of the polynomial function)
def second_derivative(x, a):
    return 2 * a

# Define the gradient descent algorithm
def gradient_descent(start_point, learning_rate, max_iterations, tolerance, a, b):
    x = start_point
    for i in range(max_iterations):
        gradient = gradient_function(x, a, b)
        if np.abs(gradient) < tolerance:
            break
        x -= learning_rate * gradient
    return x

# Set the coefficients of the polynomial function
a = 1  # Coefficient of x^2
b = 2  # Coefficient of x
c = 3  # Constant term

# Generate x values
x_values = np.linspace(-10, 10, 100)

# Generate y values using the polynomial function
y_values = polynomial_function(x_values, a, b, c)

# Find the minimum point using calculus
minimum_point = -b / (2 * a)

# Perform gradient descent from a random starting point
np.random.seed(42)
start_point = np.random.uniform(-10, 10)
learning_rate = 0.01
max_iterations = 1000
tolerance = 1e-6
minimum_found = gradient_descent(start_point, learning_rate, max_iterations, tolerance, a, b)

# Print the minimum point
print("Minimum point (calculated):", minimum_point)
print("Minimum point (found):", minimum_found)

# Plot the polynomial function and the minimum point
plt.plot(x_values, y_values, label='Polynomial Function')
plt.scatter([minimum_point], [polynomial_function(minimum_point, a, b, c)], color='red', label='Minimum Point (Calculated)')
plt.scatter([minimum_found], [polynomial_function(minimum_found, a, b, c)], color='green', label='Minimum Point (Found)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent to Find Minimum Point')
plt.legend()
plt.grid(True)
plt.show()
