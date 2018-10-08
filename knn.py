"""
The following code runs a Adaboost on 2 datasets
"""

import matplotlib.pyplot as plt
import numpy as np
from data import get_breast_cancer_data, get_balance_data
from sklearn.neighbors import KNeighborsClassifier
from time import strftime
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from plotting import plot_learning_curve, plot_knn_k
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def knn_breast_cancer():
	train_features, train_labels, test_features, test_labels = get_breast_cancer_data()
	start_time = datetime.now()
	estimator = KNeighborsClassifier()
	cross_validation = ShuffleSplit(train_size=0.8, test_size=0.2)
	k = range(1, 21, 1)
	classifier = GridSearchCV(estimator=estimator, cv=cross_validation,
		param_grid=dict(n_neighbors=k))
	classifier.fit(train_features, train_labels.ravel())
	end_time = datetime.now()
	total_time_taken = str(end_time - start_time)
	estimator = KNeighborsClassifier(n_neighbors=classifier.best_estimator_.n_neighbors)
	estimator.fit(train_features, train_labels.ravel())
	print("Score: ", classifier.score(test_features, test_labels.ravel()))
	title = 'KNN (k = %s)' % (classifier.best_estimator_.n_neighbors)
	plot_learning_curve(estimator, title,
		train_features, train_labels.ravel(), cv=cross_validation)
	plt.savefig("knn_breast_cancer.png")
	train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
	cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
	test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))
	results = classifier.cv_results_
	with open("results/knn_breast_cancer.txt", "w") as file:
		file.write("KNN with Breast Cancer Dataset\n\n")
		file.write("Optimal Number of Estimators: " + str(classifier.best_estimator_.n_neighbors))
		file.write("Grid Scores:\n\n" + str(results))
		file.write("Training Accuracy: " + str(train_accuracy) + "\n\n")
		file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n\n")
		file.write("Testing Accuracy: " + str(test_accuracy) + "\n\n")
		file.write("Total Time Taken: " + strftime(total_time_taken))

	train_accuracy = results['mean_train_score']
	test_accuracy = results['mean_test_score']
	plot_knn_k("Breast Cancer", k, train_accuracy, test_accuracy)
	plt.savefig("knn_breast_cancer_k.png")


def knn_balance_scale():
	train_features, train_labels, test_features, test_labels = get_balance_data()
	start_time = datetime.now()
	estimator = KNeighborsClassifier()
	cross_validation = ShuffleSplit(train_size=0.8, test_size=0.2)
	k = range(1, 21, 1)
	classifier = GridSearchCV(estimator=estimator, cv=cross_validation,
		param_grid=dict(n_neighbors=k))
	classifier.fit(train_features, train_labels.ravel())
	end_time = datetime.now()
	total_time_taken = str(end_time - start_time)
	estimator = KNeighborsClassifier(n_neighbors=classifier.best_estimator_.n_neighbors)
	estimator.fit(train_features, train_labels.ravel())
	print("Score: ", classifier.score(test_features, test_labels.ravel()))
	title = 'KNN (k = %s)' % (classifier.best_estimator_.n_neighbors)
	plot_learning_curve(estimator, title,
		train_features, train_labels.ravel(), cv=cross_validation)
	plt.savefig("knn_balance_scale.png")
	train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
	cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
	test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))
	results = classifier.cv_results_
	with open("results/knn_balance_scale.txt", "w") as file:
		file.write("KNN with Balance Scale Dataset\n\n")
		file.write("Optimal Number of Estimators: " + str(classifier.best_estimator_.n_neighbors))
		file.write("Grid Scores:\n\n" + str(results))
		file.write("Training Accuracy: " + str(train_accuracy) + "\n\n")
		file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n\n")
		file.write("Testing Accuracy: " + str(test_accuracy) + "\n\n")
		file.write("Total Time Taken: " + strftime(total_time_taken))

	train_accuracy = results['mean_train_score']
	test_accuracy = results['mean_test_score']
	plot_knn_k("Balance Scale", k, train_accuracy, test_accuracy)
	plt.savefig("knn_balance_scale_k.png")


knn_breast_cancer()
knn_balance_scale()

