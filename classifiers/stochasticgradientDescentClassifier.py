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
        self.acceptedPenaltyValues = [ "none", "l2", "l1", "elasticnet"]

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
        self.clf = SGDClassifier(loss="hinge", penalty="l2")
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