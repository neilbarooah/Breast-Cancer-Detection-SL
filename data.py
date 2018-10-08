import numpy as np
import os
import struct
import arff
from sklearn.model_selection import train_test_split


def get_breast_cancer_data():
	data_rows = []
	with open("wdbc.data.txt") as file:
		for line in file.readlines():
			if 'M' in line:
				data_rows.append("1," + str(line[line.find("M") + 2:]).strip())
			elif "B" in line:
				data_rows.append("2," + str(line[line.find("B") + 2:]).strip())

	data = np.loadtxt(data_rows, delimiter=",", dtype="u4")
	train_data, test_data = train_test_split(data, train_size=0.8)
	train_labels, train_features = np.hsplit(train_data, [1])
	test_labels, test_features = np.hsplit(test_data, [1])
	return train_features, train_labels, test_features, test_labels


def get_balance_data():
	data_rows = []
	with open("balance-scale.data.txt") as file:
		for line in file.readlines():
			if 'L' in line:
				data_rows.append("1" + str(line[1:]).strip())
			elif "B" in line:
				data_rows.append("2" + str(line[1:]).strip())
			elif "R" in line:
				data_rows.append("3" + str(line[1:]).strip())

	data = np.loadtxt(data_rows, delimiter=",", dtype="u4")
	train_data, test_data = train_test_split(data, train_size=0.8)
	train_labels, train_features = np.hsplit(train_data, [1])
	test_labels, test_features = np.hsplit(test_data, [1])
	return train_features, train_labels, test_features, test_labels
