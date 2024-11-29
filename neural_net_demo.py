"""
neural_net_demo.py

OCR (Optical Character Recognition) demo with a simple neural network called the multilayer perceptron classifier.
"""
# %% Imports

# Imports for machine learning
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Plotting
import matplotlib.pyplot as plt

# %% Machine Learning
# Load the digits dataset from the sklearn library
digits = load_digits()
# Set the training data (8px by 8px images) and target labels (numbers 0-9) respectively.
X, y = digits.data, digits.target

# Split the dataset into training and testing sets (usually we use a split of about 70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a neural network with the following specifications:
clf = MLPClassifier(
    hidden_layer_sizes=(100,),  # Number of neurons in each hidden layer
    max_iter=1000,  # Maximum number of iterations (converges after ~160 iterations)
    alpha=1e-4,  # Regularisation to make the model generalizable
    solver='lbfgs',  # Solver for optimized gradient descent
    verbose=10,  # Print a lot so we know what is going on
    random_state=42,  # Passing an int for the random state to ensure reproducability
    learning_rate_init=0.1  # Initial learning rate
)

# Fit the model to the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# %% Visualization
def visualize_data(X, y, num_samples=10):
    """
    Visualize the dataset with the original labels, so
    we can see if we are doing well!

    :param  X:  Matrix containing the training data (datatype list)
    :param  y:  Vector containing the target labels (datatype list)
    """
    # Create a figure
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    for i in range(num_samples):
        # Plot the values for the different pixels in the correct shape
        axes[i].imshow(X[i].reshape(8, 8), cmap='gray')
        # Add the correct label as a title
        axes[i].set_title(f"Label: {y[i]}")
        # Turn of the axis for display purposes
        axes[i].axis('off')
    plt.show()


# Visualize a few samples from the digits dataset
visualize_data(X_train, y_train)
