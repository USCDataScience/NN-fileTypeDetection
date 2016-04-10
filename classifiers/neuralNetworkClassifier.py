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

import os
import pandas as pd
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn.metrics import accuracy_score
import pickle
import logging
import sys


class NeuralNetworkClassifier:

	def __init__(self, mimetype, inputUnits, outputUnits, logging = True):
		"""
		constructor for the neural network classifier
		"""
		self.outputHeader = "#nn"
		self.inputUnits = inputUnits
		self.outputUnits = outputUnits
		self.mimetype = mimetype
		self.numberOfHiddenUnits = None
		self.nn = None
		self.testError = None

		if logging:
			self.setLoggingLevel()

	def setLoggingLevel(self):
		"""
		This function sets the logging level of the neural network
		"""
		logging.basicConfig(
            format="%(message)s",
            level=logging.INFO,
            stream=sys.stdout)

	def buildModel(self,numberOfLayers, listOfLayerSpecification, learningRate, numIters):
		"""
		This function builds the neural network layers. The input 
		is the number of layers and the second parameter contains thre
		type of layers to be used
		"""
		self.numberOfHiddenUnits = numberOfLayers
		layers = []
		for i in range(0,numberOfLayers):
			layers.append(Layer(listOfLayerSpecification[i], units = 1))
		layers.append(Layer("Softmax"))
		self.initializeNeuralNetwork(layers,learningRate,numIters)

	def initializeNeuralNetwork(self,layers,learningRate,numIters):
		"""
		This will build the Neural Network and make it ready for training
		"""
		self.nn = Classifier(
    		layers= layers,
    		learning_rate=learningRate,
    		n_iter=numIters
    	)

	def trainNetwork(self,X, Y):
		"""
		Training the neural network
		"""
		self.nn.fit(X, Y)

	def validateNetwork(self,X, Y):
		"""
		Validate the neural network
		"""
		YPred = self.nn.predict(X)
		print accuracy_score(Y, YPred)

	def testNetwork(self,X, Y):
		"""
		Test the neural network
		"""
		YPred = self.nn.predict(X)
		self.testError = 1 - accuracy_score(Y, YPred)
		print accuracy_score(Y, YPred)

	def dumpNeuralNetParameters(self, filename):
		"""
		Dump Neural Network dumpNeuralNetParameters
		"""
		nn_parameters = self.nn.get_parameters()

		with open(filename+".nnmodel", "wb") as myfile:
			myfile.write(self.outputHeader + " " + self.mimetype + " "+
				str(self.inputUnits)+" "+ str(self.numberOfHiddenUnits) + " "+
				str(self.outputUnits)+" "+str(self.testError)+"\n")

		for i in nn_parameters:
			if "hidden" in i[2]:
				for j in i[0]:
					with open(filename+".nnmodel", "a") as myfile:
						myfile.write(str(j[0]) + "\t")





