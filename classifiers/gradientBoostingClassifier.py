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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import logging

class GBClassifier:

    def __init__(self):
        """
        Inititalizes the gradient descent classifier
        """
        self.header = "#gbc"
        self.clf = None
        self.learningRate = 0.1
        self.n_estimators = 100
        self.loss = "deviance"
        self.acceptedLossValues = ["deviance", "exponential"]

    def setNumberOfEstimators(self, n_estimators):
        """
        Sets the number of estimators of Gradient Boosting Classifier
        """
        self.n_estimators = n_estimators

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

    def buildModel(self):
        """
        This builds the model of the Gradient boosting Classifier
        """
        logging.info("Building Model")
        self.clf = GradientBoostingClassifier(loss=self.loss, n_estimators=self.n_estimators,
                     learning_rate = self.learningRate)
        logging.info("Finished Building Model")

    def trainGBC(self,X, Y):
        """
        Training the Gradient Boosting Classifier
        """
        self.clf.fit(X, Y)

    def validateGBC(self,X, Y):
        """
        Validate the Gradient Boosting Classifier
        """
        YPred = self.clf.predict(X)
        print accuracy_score(Y, YPred)

    def testGBC(self,X, Y):
        """
        Test the Gradient Boosting Classifier
        """
        YPred = self.clf.predict(X)
        print accuracy_score(Y, YPred)
