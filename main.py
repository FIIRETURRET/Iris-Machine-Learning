# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:17:40 2019

@author: joutras
Tutorial: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

Classification Report: https://muthu.co/understanding-the-classification-report-in-sklearn/

Linear Discriminant Analysis(LDA): https://blog.eduonix.com/artificial-intelligence/linear-discriminant-analysis-with-scikit-learn/
Linear Discriminant Analysis(LDA): https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/

K-Nearest Neighbor(KNN): https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75
K-Nearest Neighbor(KNN): https://scikit-learn.org/stable/modules/neighbors.html

Support Vector Machine(SVM): https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

Linear Regression(LR): https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

Decision Tree Classifier(CART): http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/

Gaussian Naive Bayes(GNB): https://medium.com/@awantikdas/a-comprehensive-naive-bayes-tutorial-using-scikit-learn-f6b71ae84431
Gaussian Naive Bayes(GNB): https://scikit-learn.org/stable/modules/naive_bayes.html
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
        print("\n K-Nearest Neighbors")
        print("\nSanjay.M (Towards Data Science) - KNN (K-Nearest Neighbor) is a simple supervised classification algorithm we can use to assign a class to new data point.")
        print("\nScikit Learn Documentation - The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice.")
        input("Press Enter to continue")
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
        print("\nSupport Vector Machine")
        print("\nAvinash Navlani (Datacamp.com) - SVM constructs a hyperplane in multidimensional space to separate different classes. SVM generates optimal hyperplane in an iterative manner, which is used to minimize an error. The core idea of SVM is to find a maximum marginal hyperplane(MMH) that best divides the dataset into classes.")
        input("Press enter to continue")
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
        print("\n Linear Discriminant Analysis")
        print("\n(Eduonix.com) - LDA is a supervised algorithm which finds a subspace that maximizes the separation between features. \nThe advantage that LDA offers is that it works as a separator for classes, that is, as a classifier. However, LDA can become prone to overfitting and is vulnerable to noise/outliers.")
        print("\nScikit-Learn Documentation - A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.")
        print("\n(Eduonix.com) = In classification, LDA makes predictions by estimating the probability of a new input belonging to each class. The class that gets the highest probability is the output/predicted class.")
        input("Press Enter to continue")
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
        print("\n Linear Regression")
        print("\nNagesh Singh Chauhan (Towards Data Science) - The term “linearity” in algebra refers to a linear relationship between two or more variables. If we draw this relationship in a two-dimensional space (between two variables), we get a straight line. Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear relationship between x (input) and y(output). Hence, the name is Linear Regression. If we plot the independent variable (x) on the x-axis and dependent variable (y) on the y-axis, linear regression gives us a straight line that best fits the data points.")
        input("Press Enter to continue")
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
        print("\n Decision Tree Classifier")
        print("\nBen Alex Keen (benalexkeen.com) - A tree structure is constructed (by breaking) the dataset down into smaller subsets eventually resulting in a prediction. There are decision nodes that partition the data and leaf nodes that give the prediction that can be followed by traversing simple IF..AND..AND….THEN logic down the nodes. The root node (the first decision node) partitions the data based on the most influential feature partitioning. There are 2 measures for this, Gini Impurity and Entropy.")
        input("Press Enter to Continue")
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
                                NB - Gaussian Naive Bayes
        ###########################################################################
        """
    elif (userInput == 'NB' or userInput == 'nb' or userInput == 'Nb'):
        print("Gaussian Naive Bayes")
        print("\n Awantik Das (medium.com) - The Naive Bayes Classifier technique is based on the Bayesian theorem and is particularly suited when then high dimensional data.")
        print("\n Scikit learn Documentation - Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.")
        input("Press Enter to continue")
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