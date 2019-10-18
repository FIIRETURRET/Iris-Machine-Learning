# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:17:40 2019

@author: joutras
Tutorial: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
Classification Report: https://muthu.co/understanding-the-classification-report-in-sklearn/
Linear Discriminant Analysis(LDA): https://blog.eduonix.com/artificial-intelligence/linear-discriminant-analysis-with-scikit-learn/
Linear Discriminant Analysis(LDA): https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def measurePerformance(Y_validation, predictions):
    # Measure performance
        labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        
        print('\n\n')
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
        userInput = input("Enter to continue")
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
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
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

run = True
while (run == True):
    # Ask the user which model they would like to run
    userInput = input("Please select a model to run: (LR, LDA, KNN, CART, NB, SVM): \n")
    
    """
    ###########################################################################
                                KNN - K Nearest Neighbors
    ###########################################################################
    """
    if( userInput == 'KNN' or userInput == 'knn' or userInput =='Knn'):
        # Create model
        knn = KNeighborsClassifier()
        # Train model
        knn.fit(X_train, Y_train)
        # Make predictions using model
        predictions = knn.predict(X_validation)
        # Measure performance
        measurePerformance(Y_validation, predictions)
        
        # Release held memory
        del knn
        del predictions
        
        userInput = input("Enter to continue Q to quit: ")
        if (userInput == "Q" or userInput =="q"):
            run = False
        
        
        """
        ###########################################################################
                                    SVM - Support Vector Machines
        ###########################################################################
        """
    elif (userInput == 'SVM' or userInput == 'svm' or userInput == 'Svm'):
        # Create model
        svclassifier = SVC(kernel='linear')
        # Train model
        svclassifier.fit(X_train, Y_train)
        # Make predictions with model
        predictions = svclassifier.predict(X_validation)
        # Measure performance
        measurePerformance(Y_validation, predictions)
        
        # Free up memory
        del svclassifier
        del predictions
        
        userInput = input("Enter to continue Q to quit: ")
        if (userInput == "Q" or userInput =="q"):
            run = False
         
            
        """
        ###########################################################################
                                LDA - Linear Discriminant Analysis
        ###########################################################################
        """
    elif (userInput == 'LDA' or userInput == 'lda' or userInput == 'Lda'):
        print("\nEduonix.com - LDA is a supervised algorithm which finds a subspace that maximizes the separation between features. \nThe advantage that LDA offers is that it works as a separator for classes, that is, as a classifier. However, LDA can become prone to overfitting and is vulnerable to noise/outliers.")
        print("\nScikit-Learn Documentation - A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.")
        print("\nEduonix.com = In classification, LDA makes predictions by estimating the probability of a new input belonging to each class. The class that gets the highest probability is the output/predicted class.")
        # We need to perform feature scaling for LDA
        sc = StandardScaler()
        sc_X_train = sc.fit_transform(X_train)
        sc_X_validation = sc.transform(X_validation)
        
        # n_components refers to the number of linear discriminates that we want to retrieve.
        # n_components = 1 means we are testing performance with 1 linear discriminant.
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda_X_train = lda.fit_transform(sc_X_train, Y_train)
        lda_X_validation = lda.transform(sc_X_validation)
        
        # use random forest classifier to evaluate the performance of our LDA
        classifier = RandomForestClassifier(max_depth=2, random_state=0)
        classifier.fit(lda_X_train, Y_train)
        predictions = classifier.predict(lda_X_validation)
        measurePerformance(Y_validation, predictions)
        
        # Free up memory
        del sc
        del sc_X_train
        del sc_X_validation
        del lda
        del lda_X_train
        del lda_X_validation
        del classifier
        del predictions
        
        userInput = input("Enter to continue Q to quit: ")
        if (userInput == "Q" or userInput =="q"):
            run = False
        
        
        """
        ###########################################################################
                                LR - Linear Regression
        ###########################################################################
        """
    elif (userInput == 'LR' or userInput == 'lr' or userInput == 'Lr'):
        # make an instance of our model
        logisticRegr = LogisticRegression(solver='liblinear', multi_class='ovr')
        # Train the model
        logisticRegr.fit(X_train, Y_train)
        # Make predictions on our model
        predictions = logisticRegr.predict(X_validation)
        # Measure performance
        measurePerformance(Y_validation, predictions)
        
        # free up memory
        del logisticRegr
        del predictions
        del labels
        
        userInput = input("Enter to continue Q to quit: ")
        if (userInput == "Q" or userInput =="q"):
            run = False
      
        
        """
        ###########################################################################
                                CART - Decision Tree Classifier
        ###########################################################################
        """
    elif (userInput == 'CART' or userInput == 'cart' or userInput == 'Cart'):
        # Create a model
        cart = DecisionTreeClassifier()
        # Train the model
        cart.fit(X_train, Y_train)
        # Make a prediction with the model
        predictions = cart.predict(X_validation)
        # Measure performance
        measurePerformance(Y_validation, predictions)
        
        del cart
        del predictions
        del labels
        
        userInput = input("Enter to continue Q to quit: ")
        if (userInput == "Q" or userInput =="q"):
            run = False
            
        
        """
        ###########################################################################
                                NB - Gausiann Naive Bayes
        ###########################################################################
        """
    elif (userInput == 'NB' or userInput == 'nb' or userInput == 'Nb'):
        # Create the model
        nb = GaussianNB()
        # train the model
        nb.fit(X_train, Y_train)
        # Make predictions with the model
        predictions = nb.predict(X_validation)
        # Measure performance
        measurePerformance(Y_validation, predictions)
        
        # Free up memory
        del nb
        del predictions
        del labels
        
        userInput = input("Enter to continue Q to quit: ")
        if (userInput == "Q" or userInput =="q"):
            run = False
        
        
    elif (userInput == 'q' or userInput == 'Q' or userInput == 'Quit'):
        run = False
        
        
    else:
        print("Invalid model choice!")