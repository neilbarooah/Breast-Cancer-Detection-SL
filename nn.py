"""
The following code runs a SVM on 2 datasets
"""

import matplotlib.pyplot as plt
import numpy as np
from data import get_breast_cancer_data, get_balance_data
from sklearn.neural_network import MLPClassifier
from time import strftime
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from plotting import plot_learning_curve, plot_hidden_layer_performance
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.validation import CrossValidator
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def ann_breast_cancer():
    train_features, train_labels, test_features, test_labels = get_breast_cancer_data()
    num_layers = range(10)
    accuracy_train_layer = []
    accuracy_test_layer = []
    start_time = datetime.now()

    for layer in num_layers:
        classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
        classifier.fit(train_features, train_labels)
        accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
        accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

    layer_performance = plot_hidden_layer_performance("Breast Cancer", num_layers, accuracy_train_layer, accuracy_test_layer)
    layer_performance.savefig("nn_breast_cancer_layers_new.png")
    optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
    classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

    n_train_samples = train_features.shape[0]
    n_epochs = 8001
    n_batch = 32
    n_classes = np.unique(train_labels)

    scores_train = []
    scores_test = []
    epoch = 1
    while epoch < n_epochs:
        print('epoch: ', epoch)
        random_perm = np.random.permutation(train_features.shape[0])
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        scores_train.append(classifier.score(train_features, train_labels.ravel()))
        scores_test.append(classifier.score(test_features, test_labels.ravel()))

        epoch += 1

    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

    with open("results/ann_breast_cancer_new.txt", 'w') as file:
        file.write("ANN with Breast Cancer Dataset\n\n")
        file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n\n")
        file.write("Total Time Taken: " + strftime(total_time_taken))

    plt.figure()
    epoch_graph = range(1, 8001, 1)
    plt.plot(epoch_graph, scores_train)
    plt.plot(epoch_graph, scores_test)
    plt.legend(["Training Accuracy", "Testing Accuracy"])
    plt.title("Accuracy over epochs with " + str(optimal_num_layers) + " hidden layers")
    plt.xlabel("Epochs")
    plt.savefig("nn_breast_cancer.png")
    plt.show()


def ann_breast_compare_decay():
    train_features, train_labels, test_features, test_labels = get_breast_cancer_data()
    start_time = datetime.now()
    num_layers = range(10)
    accuracy_train_layer = []
    accuracy_test_layer = []

    for layer in num_layers:
        classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
        classifier.fit(train_features, train_labels)
        accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
        accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

    layer_performance = plot_hidden_layer_performance("Breast Cancer", num_layers, accuracy_train_layer, accuracy_test_layer)
    optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
    classifier1 = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)
    classifier2 = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-3, solver='adam', verbose=0, tol=1e-8, random_state=1)
    classifier3 = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-5, solver='adam', verbose=0, tol=1e-8, random_state=1)
    classifier4 = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-2, solver='adam', verbose=0, tol=1e-8, random_state=1)

    n_train_samples = train_features.shape[0]
    n_epochs = 101
    n_batch = 32
    n_classes = np.unique(train_labels)

    scores_train1 = []
    scores_test1 = []
    scores_train2 = []
    scores_test2 = []
    scores_train3 = []
    scores_test3 = []
    scores_train4 = []
    scores_test4 = []
    epoch = 1
    while epoch < n_epochs:
        print('epoch: ', epoch)
        random_perm = np.random.permutation(train_features.shape[0])
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            classifier1.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            classifier2.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            classifier3.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            classifier4.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        scores_train1.append(classifier1.score(train_features, train_labels.ravel()))
        scores_test1.append(classifier1.score(test_features, test_labels.ravel()))
        scores_train2.append(classifier2.score(train_features, train_labels.ravel()))
        scores_test2.append(classifier2.score(test_features, test_labels.ravel()))
        scores_train3.append(classifier3.score(train_features, train_labels.ravel()))
        scores_test3.append(classifier3.score(test_features, test_labels.ravel()))
        scores_train4.append(classifier4.score(train_features, train_labels.ravel()))
        scores_test4.append(classifier4.score(test_features, test_labels.ravel()))

        epoch += 1

    plt.figure()
    epoch_graph = range(1, 101, 1)
    plt.plot(epoch_graph, scores_test3)
    plt.plot(epoch_graph, scores_test1)
    plt.plot(epoch_graph, scores_test2)
    plt.plot(epoch_graph, scores_test4)
    plt.legend(["Testing Accuracy (0.00001)", "Testing Accuracy (0.0001)", "Testing Accuracy (0.001)", "Testing Accuracy (0.01)"])
    plt.title("Accuracy over epochs with " + str(optimal_num_layers) + " hidden layers")
    plt.xlabel("Epochs")
    plt.show()


def ann_balance_scale():
    train_features, train_labels, test_features, test_labels = get_balance_data()
    start_time = datetime.now()
    num_layers = range(10)
    accuracy_train_layer = []
    accuracy_test_layer = []

    for layer in num_layers:
        classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600, alpha=0.001)
        classifier.fit(train_features, train_labels)
        accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
        accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

    layer_performance = plot_hidden_layer_performance("Balance Scale", num_layers, accuracy_train_layer, accuracy_test_layer)
    layer_performance.savefig("nn_balance_layers.png")
    optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
    classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

    n_train_samples = train_features.shape[0]
    n_epochs = 101
    n_batch = 32
    n_classes = np.unique(train_labels)

    scores_train = []
    scores_test = []
    epoch = 1
    while epoch < n_epochs:
        print('epoch: ', epoch)
        random_perm = np.random.permutation(train_features.shape[0])
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        scores_train.append(classifier.score(train_features, train_labels.ravel()))
        scores_test.append(classifier.score(test_features, test_labels.ravel()))

        epoch += 1

    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

    with open("results/ann_balance_scale.txt", 'w') as file:
        file.write("ANN with Balance Scale Dataset\n\n")
        file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n\n")
        file.write("Total Time Taken: " + strftime(total_time_taken))

    plt.figure()
    epoch_graph = range(1, 101, 1)
    plt.plot(epoch_graph, scores_train)
    plt.plot(epoch_graph, scores_test)
    plt.legend(["Training Accuracy", "Testing Accuracy"])
    plt.title("Accuracy over epochs with " + str(optimal_num_layers) + " hidden layers")
    plt.xlabel("Epochs")
    plt.savefig("nn_balance_scale.png")
    plt.show()


def compare_l2_regularization():
    train_features, train_labels, test_features, test_labels = get_breast_cancer_data()
    optimal_num_layers = 6
    num_neurons = [optimal_num_layers * [16]]
    start_time = datetime.now()
    train_accuracy1 = []
    test_accuracy1 = []
    train_accuracy2 = []
    test_accuracy2 = []
    iterations = range(250)
    nn1 = buildNetwork(30, 16, 1, bias=True)
    nn2 = buildNetwork(30, 16, 1, bias=True)
    dataset = ClassificationDataSet(len(train_features[0]), len(train_labels[0]), class_labels=["1", "2"])
    
    for instance in range(len(train_features)):
        dataset.addSample(train_features[instance], train_labels[instance])
    
    trainer1 = BackpropTrainer(nn1, dataset, weightdecay=0.0001)
    validator1 = CrossValidator(trainer1, dataset)
    print(validator1.validate())
    
    trainer2 = BackpropTrainer(nn2, dataset, weightdecay=0.001)
    validator2 = CrossValidator(trainer2, dataset)
    print(validator2.validate())

    for iteration in iterations:
        train_accuracy1.append(sum((np.array([np.round(nn1.activate(test)) for test in train_features]) - train_labels)**2)/float(len(train_labels)))
        test_accuracy1.append(sum((np.array([np.round(nn1.activate(test)) for test in test_features]) - test_labels)**2)/float(len(test_labels)))
        train_accuracy2.append(sum((np.array([np.round(nn2.activate(test)) for test in train_features]) - train_labels)**2)/float(len(train_labels)))
        test_accuracy2.append(sum((np.array([np.round(nn2.activate(test)) for test in test_features]) - test_labels)**2)/float(len(test_labels)))

    plt.plot(iterations, train_accuracy1)
    plt.plot(iterations, test_accuracy1)
    plt.plot(iterations, train_accuracy2)
    plt.plot(iterations, test_accuracy2)
    plt.legend(["Train Accuracy (0.0001)", "Test Accuracy (0.0001)", "Train Accuracy (0.001)", "Test Accuracy (0.001"])
    plt.xlabel("Num Epoch")
    plt.ylabel("Percent Error")
    plt.title("Neural Network on Breast Cancer Data with " + str(num_neurons) + " layers")
    plt.savefig("nn_breast_cancer_weight_decay.png")

ann_breast_cancer()
#ann_balance_scale()
