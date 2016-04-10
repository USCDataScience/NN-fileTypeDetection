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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNNClassifier:
	
	def __init__(self):
		"""
		This is the constructor for the KNN Classifier
		"""
		self.outputHeader = "#knn"
		self.clf = None
		self.n_neighbors = 5
		self.weights = "uniform"
		self.algorithm = "auto"

	def buildModel(self):
		"""
		This builds the model of the KNN Classifier
		"""
		self.clf = KNeighborsClassifier(n_neighbors=self.n_neighbors,
					 	weights = self.weights, algorithm=self.algorithm)

	def setNeighbors(self, param):
		"""
		This sets the n neighbor for the KNN Classifier.
		"""
		self.n_neighbors = param

	def setAlgorithm(self, param):
		"""
		This sets the algorithm parameter for the KNN Classifier
		"""
		if param in ["auto", "ball_tree", "kd_tree", "brute"]:
			self.algorithm = param
		else:
			print "unknown parameter defaulting to auto."

	def setWeights(self, param):
		"""
		This sets the weights parameter for KNN Classifier 
		"""
		self.weights = param

	def trainKNN(self,X, Y):
		"""
		Training the KNN Classifier
		"""
		self.clf.fit(X, Y)

	def validateKNN(self,X, Y):
		"""
		Validate the KNN Classifier
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)

	def testKNN(self,X, Y):
		"""
		Test the KNN Classifier
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)
