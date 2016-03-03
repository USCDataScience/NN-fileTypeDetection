import os
import pandas as pd
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn.metrics import accuracy_score
import pickle

# set the environment variable for theano library
train_dir = "train.csv"
test_dir = "test.csv"
validation_dir = "val.csv"

XTrain = None
YTrain = None
XTest = None
YTest = None
nn = None


## default parameters, change as per need
fileHeader = "#nn"
mimeType = "application/x-grib"
numberOfInputs = 256 # this is the bytes ranging from 0..256
numberOfOutputs = 1
numberOfHiddenLayers = 2
testError = 0.0208829625450076 ## hard coding now, needs to be computed of accuracy

def readData():
	"""
	Reads data from CSV
	"""
	loadTrainingData()
	loadTestData()
	modelBuilder()
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
	global nn
	y_results = nn.predict(XTest)
	print accuracy_score(YTest, y_results)

def modelBuilder():
	global nn, XTrain, YTrain, XTest, YTest
	nn = Classifier(
    layers=[
        Layer("Sigmoid", units=1, pieces=3),
        Layer("Sigmoid", units=1, pieces=3),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=25)
	nn.fit(XTrain, YTrain)
	dumpNeuralNetworkParameters()
	
def dumpNeuralNetworkParameters():
	global nn, fileHeader, testError, numberOfInputs, numberOfHiddenLayers, numberOfOutputs

	nn_parameters = nn.get_parameters()
	with open("test-example.nnmodel", "a") as myfile:
		myfile.write(fileHeader+" "+mimeType+" "+str(numberOfInputs)+" "+
			str(numberOfHiddenLayers) + " "+str(numberOfOutputs)+" "+str(testError)+"\n")
	for i in nn_parameters:
		if "hidden" in i[2]:
			for j in i[0]:
				with open("test-example.nnmodel", "a") as myfile:
					myfile.write(str(j[0])+"\t")


if __name__ == '__main__':
	readData()