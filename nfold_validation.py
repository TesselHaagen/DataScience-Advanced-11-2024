"""
nfold_validation

Apply n-fold validation to a machine learning model. This is a method
where we use different train/test splits to get a more reliable accuracy.
In this case we classify the Iris dataset 5 times with different splits.
"""

#%% Imports
# Dataset from Scikit Learn
from sklearn.datasets import load_iris

# Clustering algorithm
from sklearn.neighbors import KNeighborsClassifier

# Cross-validation helper methods
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

#%% Configuration
# Number for the n-fold validation
n = 5

#%% Load dataset
iris = load_iris()

#%% Pre-processing
# Use this if you used the load_iris function
X, y = iris.data, iris.target

#%% Clustering
model = KNeighborsClassifier()

# Scikit Learn will make different splits of the data for us and return a 
# list of n accuracy scores.
nscores = cross_val_score(model, X, y, cv=n)
print(nscores)
print(nscores.mean())

#%% Leave One Out cross-validation
looscores = cross_val_score(model, X, y, cv=LeaveOneOut())
print(looscores)
print(looscores.mean())
