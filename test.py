# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:14:58 2019

@author: joutras
https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
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
print('accuracy score:')
print(accuracy_score(Y_validation, predictions))
print('confusion matrix:')
print(confusion_matrix(Y_validation, predictions))
print('classification report:')
print(classification_report(Y_validation, predictions))
