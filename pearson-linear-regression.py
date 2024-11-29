import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the Pearson's father and son dataset
# Note that using Pandas here would have made more sense, but you can use
data = np.genfromtxt('../../data/pearson-father-son.csv', delimiter=',', skip_header=1)

# Extract father's height (feature) and son's height (target)
X = data[:, 1].reshape(-1, 1)  # Reshape to a column vector
y = data[:, 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot the data points and the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear regression')
plt.xlabel("Father's Height")
plt.ylabel("Son's Height")
plt.title("Linear Regression: Father's Height vs Son's Height")
plt.legend()
plt.show()
