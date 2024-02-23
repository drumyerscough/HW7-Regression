"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler


def test_prediction():
	"""
	Creates an instance of the LogisticRegressor class, sets the weights, and tests that the make_prediction() method returns the correct value for an arbitrary array of input features.
	"""
	log_model = logreg.LogisticRegressor(num_feats=6)
	log_model.W = np.array([1,2,1,2,0.3,0.5])
	X = np.array([[0.1,0.2,0.05,0.9,0.9,0.7]])
	assert round(log_model.make_prediction(X), 5) == 0.95120


def test_loss_function():
	"""
	Creates an instance of the LogisticRegressor class and tests that the loss_function() method returns the correct value for an arbitrary sequence of labels and predictions.
	"""
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.05, tol=0.0000001, max_iter=100, batch_size=5)
	assert round(log_model.loss_function([0,0,0,1,1,1], [0.1,0.2,0.05,0.9,0.9,0.7]), 5) == 0.15787


def test_gradient():
	"""
	Creates an instance of the LogisticRegressor class, sets the weights, and tests that the calculate_gradient() method returns the correct value for an arbitrary array of input features.
	"""
	log_model = logreg.LogisticRegressor(num_feats=6)
	log_model.W = np.array([1,2,1,2,0.3,0.5])
	y_true = np.array([0])
	X = np.array([[0.1,0.2,0.05,0.9,0.9,0.7]])
	assert round(log_model.calculate_gradient(y_true, X), 5) == 0.09512


def test_training():
	"""
	Creates an instance of the LogisticRegressor class, runs on a test dataset, and tests that the training loss changes across iterations.
	"""
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
			],
		split_percent=0.8,
		split_seed=42
	)
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.05, tol=0.0000001, max_iter=100, batch_size=5)
	log_model.train_model(X_train, y_train, X_val, y_val)
	
	assert len(log_model.loss_hist_train.unique) > 1