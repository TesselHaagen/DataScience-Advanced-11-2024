"""
german_credit_from_dataset.py

Note: this dataset was imported with some helper code, since the columns were not clearly labeled.
While there are .csv versions of the dataset available, one easy to find example actually contained an example
of wrong label encoding. The dataset has been wrangled into submission for usage in this code.
"""
import pandas as pd
from numpy import ravel
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data.import_german_credit import fetch_data

# Fetch the data
X, y = fetch_data()

# EDA
#print(X.info())
#print(X.head())

# Feature engineering
# Select the categorical columns
cat = X.select_dtypes(include=['object']).columns

# Create the dummies
dummies = pd.get_dummies(X, columns=cat)

# Drop the original columns and attach the new columns
X = X.drop(columns=cat)
X = pd.concat([X, dummies], axis=1)

# Classification models we can apply
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

for classifier in classifiers:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    res = classifier.fit(X_train, ravel(y_train))
    y_pred = classifier.predict(X_test)

    print(f"Classifier: {classifier}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")

    # Print the classifier summary if we have one
    if hasattr(res, 'summary'): print(res.summary)
