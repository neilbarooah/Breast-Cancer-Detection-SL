"""
The following code runs a SVM on 2 datasets
"""

import matplotlib.pyplot as plt
import numpy as np
from data import get_breast_cancer_data, get_balance_data
from sklearn.svm import SVC
from time import strftime
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from plotting import plot_learning_curve, plot_svm_performance
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def svm_breast_cancer():
    train_features, train_labels, test_features, test_labels = get_breast_cancer_data()
    start_time = datetime.now()
    estimator = SVC()
    cross_validation = ShuffleSplit(train_size=0.8, test_size=0.2)
    kernels = ["linear", "sigmoid", "rbf"]
    classifier = GridSearchCV(
        estimator=estimator,
        cv=cross_validation,
        param_grid=dict(kernel=kernels))

    classifier.fit(train_features, train_labels.ravel())
    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    title = 'SVM (kernel = %s)' % (classifier.best_estimator_.kernel)
    estimator = SVC(kernel=classifier.best_estimator_.kernel)
    estimator.fit(train_features, train_labels.ravel())
    print("Score: ", classifier.score(test_features, test_labels.ravel()))
    plot_learning_curve(estimator, title, train_features, train_labels.ravel(), cv=cross_validation)
    plt.savefig('svm_breast_cancer.png')
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))
    results = classifier.cv_results_
    with open("results/svm_breast_cancer.txt", 'w') as file:
    	file.write("SVM with Breast Cancer Dataset\n\n")
    	file.write("Best Kernel: " + str(classifier.best_estimator_.kernel) + "\n\n")
    	file.write("Grid Scores:\n\n" + str(results))
    	file.write("Training Accuracy: " + str(train_accuracy) + "\n\n")
    	file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n\n")
    	file.write("Testing Accuracy: " + str(test_accuracy) + "\n\n")
    	file.write("Total Time Taken: " + strftime(total_time_taken))

    train_accuracy = results['mean_train_score']
    test_accuracy = results['mean_test_score']

    plot_svm_performance("Breast Cancer", kernels, train_accuracy, test_accuracy)
    plt.savefig("svm_breast_cancer_kernels.png")



def svm_balance_scale():
    train_features, train_labels, test_features, test_labels = get_balance_data()
    start_time = datetime.now()
    estimator = SVC()
    cross_validation = ShuffleSplit(train_size=0.8, test_size=0.2)
    kernels = ["linear", "sigmoid", "rbf"]
    classifier = GridSearchCV(
        estimator=estimator,
        cv=cross_validation,
        param_grid=dict(kernel=kernels))

    classifier.fit(train_features, train_labels.ravel())
    title = 'SVM (kernel = %s)' % (classifier.best_estimator_.kernel)
    estimator = SVC(kernel=classifier.best_estimator_.kernel)
    estimator.fit(train_features, train_labels.ravel())
    print("Score: ", classifier.score(test_features, test_labels.ravel()))
    plot_learning_curve(estimator, title, train_features, train_labels.ravel(), cv=cross_validation)
    plt.savefig('svm_balance_scale.png')
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))
    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    results = classifier.cv_results_
    with open("results/svm_balance_scale.txt", 'w') as file:
    	file.write("SVM with Balance Scale Dataset\n\n")
    	file.write("Best Kernel: " + str(classifier.best_estimator_.kernel) + "\n\n")
    	file.write("Grid Scores:\n\n" + str(results))
    	file.write("Training Accuracy: " + str(train_accuracy) + "\n\n")
    	file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n\n")
    	file.write("Testing Accuracy: " + str(test_accuracy) + "\n\n")
    	file.write("Total Time Taken: " + strftime(total_time_taken))

    train_accuracy = results['mean_train_score']
    test_accuracy = results['mean_test_score']

    plot_svm_performance("Balance Scale", kernels, train_accuracy, test_accuracy)
    plt.savefig("svm_balance_scale_kernels.png")


svm_breast_cancer()
svm_balance_scale()