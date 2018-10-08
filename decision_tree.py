"""
The following code runs a Decision Tree Classifier from sklearn on 2 datasets
"""

import matplotlib.pyplot as plt
import numpy as np
from data import get_breast_cancer_data, get_balance_data
from sklearn.tree import DecisionTreeClassifier
from time import strftime
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from plotting import plot_learning_curve, plot_dtree_depth_performance
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def decision_tree_breast_cancer():
    # Use grid search to test max_depth from 1 to 100.
    train_features, train_labels, test_features, test_labels = get_breast_cancer_data()
    start_time = datetime.now()
    estimator = DecisionTreeClassifier()
    cross_validation = ShuffleSplit(train_size=0.8, test_size=0.2)
    max_depth = range(1, 100, 1)
    classifier = GridSearchCV(
        estimator=estimator,
        cv=cross_validation,
        param_grid=dict(max_depth=max_depth))
    classifier.fit(train_features, train_labels.ravel())
    end_time = datetime.now()
    title = 'Decision Tree (max_depth = %s)' % (classifier.best_estimator_.max_depth)
    plot_learning_curve(estimator, title, train_features, train_labels.ravel(), cv=cross_validation)
    plt.savefig('dtree_breast_cancer_trial_1.png')

    # Test performance on the optimal max depth
    optimal_depth = classifier.best_estimator_.max_depth
    estimator = DecisionTreeClassifier(max_depth=optimal_depth)
    estimator.fit(train_features, train_labels.ravel())
    total_time_taken = str(end_time - start_time)
    score = classifier.score(test_features, test_labels.ravel())
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))
    results = classifier.cv_results_
    with open("results/dtree_breast_cancer.txt", 'w') as file:
        file.write("Decision Tree with Breast Cancer Dataset\n\n")
        file.write("Optimal Depth: " + str(optimal_depth) + "\n\n")
        file.write("CV Results:\n\n" + str(results) + "\n\n")
        file.write("Feature Importance: " + str(estimator.feature_importances_) + "\n\n")
        file.write("Score: " + str(score) + "\n\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n\n")
        file.write("Total Time Taken: " + strftime(total_time_taken))

    train_accuracy = results['mean_train_score']
    test_accuracy = results['mean_test_score']

    plot_dtree_depth_performance("Breast Cancer", max_depth, train_accuracy, test_accuracy)
    plt.savefig("dtree_breast_cancer_depth.png")


def decision_tree_balance_scale():
    train_features, train_labels, test_features, test_labels = get_balance_data()
    start_time = datetime.now()
    estimator = DecisionTreeClassifier()
    cross_validation = ShuffleSplit(train_size=0.8, test_size=0.2)
    max_depth = range(1, 100, 1)
    classifier = GridSearchCV(
        estimator=estimator,
        cv=cross_validation,
        param_grid=dict(max_depth=max_depth))
    classifier.fit(train_features, train_labels.ravel())
    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    title = 'Decision Tree (max_depth = %s)' % (classifier.best_estimator_.max_depth)
    plot_learning_curve(estimator, title, train_features, train_labels.ravel(), cv=cross_validation)
    plt.savefig('dtree_balance_scale.png')
    estimator = DecisionTreeClassifier(max_depth=classifier.best_estimator_.max_depth)
    estimator.fit(train_features, train_labels.ravel())
    score = classifier.score(test_features, test_labels.ravel())
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))
    results = classifier.cv_results_
    with open("results/dtree_balance_scale.txt", 'w') as file:
        file.write("Decision Tree with Balance Scale Dataset\n\n")
        file.write("Best Depth: " + str(classifier.best_estimator_.max_depth) + "\n\n")
        file.write("Grid Scores:\n\n" + str(results) + "\n\n")
        file.write("Feature Importance: " + str(estimator.feature_importances_) + "\n\n")
        file.write("Score: " + str(score) + "\n\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n\n")
        file.write("Total Time Taken: " + strftime(total_time_taken))

    train_accuracy = results['mean_train_score']
    test_accuracy = results['mean_test_score']

    plot_dtree_depth_performance("Balance Scale", max_depth, train_accuracy, test_accuracy)
    plt.savefig("dtree_balance_scale_depth.png")


decision_tree_breast_cancer()
decision_tree_balance_scale()