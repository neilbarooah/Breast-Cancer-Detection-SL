"""
The following code runs a Adaboost on 2 datasets
"""

import matplotlib.pyplot as plt
import numpy as np
from data import get_breast_cancer_data, get_balance_data
from sklearn.ensemble import AdaBoostClassifier
from time import strftime
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from plotting import plot_learning_curve, plot_boosting_performance
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def boosting_breast_cancer():
	train_features, train_labels, test_features, test_labels = get_breast_cancer_data()
	start_time = datetime.now()
	estimator = AdaBoostClassifier()
	cross_validation = ShuffleSplit(train_size=0.8, test_size=0.2)
	num_estimators = range(1, 100, 1)
	classifier = GridSearchCV(estimator=estimator, cv=cross_validation,
		param_grid=dict(n_estimators=num_estimators))
	classifier.fit(train_features, train_labels.ravel())
	end_time = datetime.now()
	total_time_taken = str(end_time - start_time)
	estimator = AdaBoostClassifier(n_estimators=classifier.best_estimator_.n_estimators)
	estimator.fit(train_features, train_labels.ravel())
	print("Score: ", classifier.score(test_features, test_labels.ravel()))
	title = "Ada Boost (num_estimators = %s)" % str(classifier.best_estimator_.n_estimators)
	plot_learning_curve(estimator, title,
		train_features, train_labels.ravel(), cv=cross_validation)
	plt.savefig("adaboost_breast_cancer.png")
	train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
	cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
	test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))
	results = classifier.cv_results_
	with open("results/adaboost_breast_cancer.txt", "w") as file:
		file.write("Adaboost with Breast Cancer Dataset\n\n")
		file.write("Optimal Number of Estimators: " + str(classifier.best_estimator_.n_estimators))
		file.write("Grid Scores:\n\n" + str(results))
		file.write("Training Accuracy: " + str(train_accuracy) + "\n\n")
		file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n\n")
		file.write("Testing Accuracy: " + str(test_accuracy) + "\n\n")
		file.write("Total Time Taken: " + strftime(total_time_taken))

	train_accuracy = results['mean_train_score']
	test_accuracy = results['mean_test_score']

	plot_boosting_performance("Breast Cancer", num_estimators, train_accuracy, test_accuracy)
	plt.savefig("adaboost_breast_cancer_num_estimators.png")



def boosting_balance_scale():
	train_features, train_labels, test_features, test_labels = get_balance_data()
	start_time = datetime.now()
	estimator = AdaBoostClassifier()
	cross_validation = ShuffleSplit(train_size=0.8, test_size=0.2)
	num_estimators = range(1, 100, 1)
	classifier = GridSearchCV(estimator=estimator, cv=cross_validation,
		param_grid=dict(n_estimators=num_estimators))
	classifier.fit(train_features, train_labels.ravel())
	end_time = datetime.now()
	total_time_taken = str(end_time - start_time)
	estimator = AdaBoostClassifier(n_estimators=classifier.best_estimator_.n_estimators)
	estimator.fit(train_features, train_labels.ravel())
	print("Score: ", classifier.score(test_features, test_labels.ravel()))
	title = "Ada Boost (num_estimators = %s)" % str(classifier.best_estimator_.n_estimators)
	plot_learning_curve(estimator, title,
		train_features, train_labels.ravel(), cv=cross_validation)
	plt.savefig("adaboost_balance_scale.png")
	train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
	cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
	test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))
	results = classifier.cv_results_
	with open("results/adaboost_balance_scale.txt", "w") as file:
		file.write("Adaboost with Balance Scale Dataset\n\n")
		file.write("Optimal Number of Estimators: " + str(classifier.best_estimator_.n_estimators))
		file.write("Grid Scores:\n\n" + str(results))
		file.write("Training Accuracy: " + str(train_accuracy) + "\n\n")
		file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n\n")
		file.write("Testing Accuracy: " + str(test_accuracy) + "\n\n")
		file.write("Total Time Taken: " + strftime(total_time_taken))

	train_accuracy = results['mean_train_score']
	test_accuracy = results['mean_test_score']

	plot_boosting_performance("Balance Scale", num_estimators, train_accuracy, test_accuracy)
	plt.savefig("adaboost_balance_num_estimators.png")


boosting_breast_cancer()
boosting_balance_scale()

