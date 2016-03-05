import pandas as pd
import numpy as np
from classifiers.neuralNetworkClassifier import NeuralNetworkClassifier

XTrain = None
YTrain = None
XTest = None
YTest = None

train_dir = "dataset/xgrib/train.csv"
test_dir = "dataset/xgrib/test.csv"
validation_dir = "dataset/xgrib/val.csv"


def readData():
	"""
	Reads data from CSV
	"""
	loadTrainingData()
	loadTestData()
	testNeuralNetwork()

def loadTrainingData():
	"""
	This function loads training data for Theano to train
	Output: Updates XTrain and YTrain for trainign neural network using Theano
	"""
	global XTrain, YTrain
	df = pd.read_csv(train_dir).as_matrix()
	XTrain = df[:,:255]
	YTrain = np.int_(df[:,256])

def loadTestData():
	"""
	This function loads testing data for Theano to test.
	Output: Updates XTest and YTest for testing neural network
	"""
	global XTest, YTest
	df = pd.read_csv(test_dir).as_matrix()
	XTest = df[:,:255]
	YTest = np.int_(df[:,256])

def testNeuralNetwork():
	"""
	This function performs the training and outputs the accuracy of the neural network.
	"""
	layers = ["Sigmoid", "Sigmoid"]
	tester = NeuralNetworkClassifier("application/x-grib",256,1)
	tester.buildModel(len(layers), layers, 0.001, 25)
	tester.trainNetwork(XTrain, YTrain)
	tester.testNetwork(XTest, YTest)

	tester.dumpNeuralNetParameters("x-grib")

if __name__ == '__main__':
	readData()
