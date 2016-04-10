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

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import logging

class GDClassifier:

    def __init__(self):
        """
        Inititalizes the gradient descent classifier
        """
        self.header = "#gdc"
        self.clf = None
        self.loss = "hinge" #default value
        self.penalty = "l2" #deafult value
        self.acceptedLossValues = ["hinge", "log", "modified_huber",
         "squared_hinge", "perceptron"]
        self.n_iter = None
        self.acceptedPenaltyValues = [ "none", "l2", "l1", "elasticnet"]

    def setNumberOfIteration(self, n_iters):
        """
        Sets the number of iteration of Gradient Descent Classifier
        """
        self.n_iter = n_iters

    def setLoss(self, loss):
        """
        Sets the loss parameter for the SGDC
        """
        try:
            if loss in self.acceptedLossValues:
                self.loss = loss
            else:
                raise ValueError("Error in input value")
        except Exception as error:
            logging.warning("Error: No such loss value:%s", loss)

    def setPenalty(self, penalty):
        """
        Sets the penalty for the SGDC classifier. 
        This is the regularization value
        """
        try:
            if penalty in self.acceptedPenaltyValues:
                self.penalty = penalty
            else:
                raise ValueError("Error in input value")
        except Exception as error:
            logging.warning("Error: no such penalty value:%s", penalty)

    def buildModel(self):
        """
        This builds the model of the Gradient Descent Classifier
        """
        logging.info("Building Model")
        self.clf = SGDClassifier(loss=self.loss, penalty=self.penalty, n_iter = self.n_iter)
        logging.info("Finished Building Model")

    def trainGDC(self,X, Y):
        """
        Training the Gradient Descent Classifier
        """
        self.clf.fit(X, Y)

    def validateGDC(self,X, Y):
        """
        Validate the Gradient Descent Classifier
        """
        YPred = self.clf.predict(X)
        print accuracy_score(Y, YPred)

    def testGDC(self,X, Y):
        """
        Test the Gradient Descent Classifier
        """
        YPred = self.clf.predict(X)
        print accuracy_score(Y, YPred)
