from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

class GDClassifier:

    def __init__(self):
        """
        Inititalizes the gradient descent classifier
        """
        self.header = "#gdc"
        self.clf = None

    def buildModel(self):
        """
        This builds the model of the Gradient Descent Classifier
        """
        self.clf = SGDClassifier(loss="hinge", penalty="l2")

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