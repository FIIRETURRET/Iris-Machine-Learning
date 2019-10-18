# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:14:58 2019

@author: joutras
Tutorial: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
Classification Report: https://muthu.co/understanding-the-classification-report-in-sklearn/
Linear Discriminant Analysis(LDA): https://blog.eduonix.com/artificial-intelligence/linear-discriminant-analysis-with-scikit-learn/
"""

import pandas 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "C:\\Users\\joutras\\Documents\\Python Scripts\\Iris\\iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# Shape
print(dataset.shape, "\n\n")
#head
print(dataset.head(20), "\n\n")
# descriptions
print(dataset.describe(), "\n\n")
# class distribution
print(dataset.groupby('class').size(), "\n\n")


# Univariate Plots
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histogram
dataset.hist()
plt.show()

# Multivariate Plots
# scatter plot matrix
scatter_matrix(dataset)
plt.show()
print("\n\n")


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state=seed)


# 10-fold cross validation
# split our dataset into 10 parts, train on 9 and test on 1
# repeat this process for all combinations of train-test splits.
# Test options and evaluation metric
seed = 7
scoring = 'accuracy' # correctly predicted instances / total number of instances.

# Spot Check Algorithms
models = []
models.append(('LF', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: mean:%f std:(%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

print('accuracy score:')
print(accuracy_score(Y_validation, predictions))
print("\n")
print('confusion matrix:')
cm = confusion_matrix(Y_validation, predictions, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print("\n")
print('classification report:')
print('Percision - What percent of your predictions were correct')
print('Precision is the ability of a classifier not to label an instance positive that is actually negative. For each class it is defined as the ratio of true positives to the sum of true and false positives.')
print('\n')
print('Recall - What percent of the positive casses did you catch?')
print('Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives.')
print('\n')
print('F1 Score - What percent of positive predictions were correct?')
print('The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy.')
print('\n')
print(classification_report(Y_validation, predictions))
print("\n\n")


# Try using SVM
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)
predictions = svclassifier.predict(X_validation)

print('accuracy score:')
print(accuracy_score(Y_validation, predictions))
print("\n")
print('confusion matrix:')
cm = confusion_matrix(Y_validation, predictions, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print("\n")
print('classification report:')
print('Percision - What percent of your predictions were correct')
print('Precision is the ability of a classifier not to label an instance positive that is actually negative. For each class it is defined as the ratio of true positives to the sum of true and false positives.')
print('\n')
print('Recall - What percent of the positive casses did you catch?')
print('Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives.')
print('\n')
print('F1 Score - What percent of positive predictions were correct?')
print('The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy.')
print('\n')
print(classification_report(Y_validation, predictions))
print("\n\n")


# Try using LDA
from sklearn import datasets
iris = datasets.load_iris()
iris_features = iris.data
print(iris_features)
iris_target = iris.target
print(iris_target)
print(X_validation)
print(X_train)
print(Y_train)