"""
svm_demo.py

SVM demo on the breast cancer dataset.

Source: https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
"""
# %% Imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# %% Data loading
cancer = datasets.load_breast_cancer()

# %% Data exploration
# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)

# %% Split data
# We skip the preparation step for this specific example.
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data,
    cancer.target,
    test_size=0.3,
    random_state=109
)

# %% Fit model
# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# %% Model evaluation
# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Create the confusion matrix to see the true negatives, false positives, false negatives, true positives
# (in that order, top to bottom left to right)
conf = metrics.confusion_matrix(y_test, y_pred)
print(conf)