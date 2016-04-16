#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
#

import pandas as pd
import numpy as np
from classifiers.neuralNetworkClassifier import NeuralNetworkClassifier
from classifiers.decisionTreeClassifier import DecisionTreeClassifier
from classifiers.supportVectorMachine import SupportVectorMachineClassifier
from classifiers.gaussianNB import GaussianNBClassifier
from classifiers.randomForestClassifier import RFClassifier
from classifiers.kNeighborhoodClassifier import KNNClassifier
from classifiers.stochasticgradientDescentClassifier import GDClassifier
from classifiers.gradientBoostingClassifier import GBClassifier


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
	#testRF()
	#testKNN()
	#testGDC()
	testGBC()


def loadTrainingData():
	"""
	This function loads training data for Theano to train
	Output: Updates XTrain and YTrain for trainign neural network using Theano
	"""
	global XTrain, YTrain
	df = pd.read_csv(train_dir).as_matrix()
	XTrain = df[:,:256]
	YTrain = np.int_(df[:,256])

def loadTestData():
	"""
	This function loads testing data for Theano to test.
	Output: Updates XTest and YTest for testing neural network
	"""
	global XTest, YTest
	df = pd.read_csv(test_dir).as_matrix()
	XTest = df[:,:256]
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
	tester.pickleClassifier()
	tester.testTree(XTest, YTest)

def testSVM():
	tester = SupportVectorMachineClassifier()
	tester.tuneParameter()
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

def testKNN():
	tester = KNNClassifier()
	tester.buildModel()
	tester.trainKNN(XTrain,YTrain)
	tester.testKNN(XTest, YTest)

def testGDC():
	tester = GDClassifier()
	tester.buildModel()
	tester.trainGDC(XTrain,YTrain)
	tester.testGDC(XTest, YTest)

def testGBC():
	tester = GBClassifier()
	tester.buildModel()
	tester.trainGBC(XTrain,YTrain)
	tester.testGBC(XTest, YTest)



if __name__ == '__main__':
	readData()
