import pandas as pd
import numpy as np
from classifiers.neuralNetworkClassifier import NeuralNetworkClassifier
from classifiers.decisionTreeClassifier import DecisionTreeClassifier
from classifiers.supportVectorMachine import SupportVectorMachineClassifier
from classifiers.gaussianNB import GaussianNBClassifier
from classifiers.randomForestClassifier import RFClassifier

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
	#testNeuralNetwork()
	#testDecisionTree()
	#testSVM()
	#testGaussianNB()
	testRF()

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

def testDecisionTree():
	tester = DecisionTreeClassifier()
	tester.buildModel()
	tester.trainTree(XTrain, YTrain)
	tester.testTree(XTest, YTest)

def testSVM():
	tester = SupportVectorMachineClassifier()
	tester.buildModel()
	tester.trainSVM(XTrain,YTrain)
	tester.testSVM(XTest, YTest)

def testGaussianNB():
	tester = GaussianNBClassifier()
	tester.buildModel()
	tester.trainGaussianNB(XTrain,YTrain)
	tester.testGaussianNB(XTest, YTest)

def testRF():
	tester = RFClassifier()
	tester.buildModel()
	tester.trainRF(XTrain,YTrain)
	tester.testRF(XTest, YTest)

if __name__ == '__main__':
	readData()
