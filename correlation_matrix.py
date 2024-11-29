"""
correlation_matrix.py

Plots a correlation matrix (using Seaborn's heatmap) to indicate
the relation between the numerical columns of the Iris dataset.
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = sns.load_dataset("iris")

# Plot correlation matrix
correlation_matrix = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()
