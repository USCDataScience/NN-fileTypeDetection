from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

class DecisionTreeClassifier:
	def __init__(self):
		"""
		This is the constructor for the decision tree classifier
		"""
		self.outputHeader = "#dt"
		self.clf = None

	def buildModel(self):
		"""
		This builds the model of the classifier
		"""
		self.clf = tree.DecisionTreeClassifier()

	def trainTree(self,X, Y):
		"""
		Training the neural network
		"""

		self.clf.fit(X, Y)

	def validateTree(self,X, Y):
		"""
		Validate the neural network
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)

	def testTree(self,X, Y):
		"""
		Test the neural network
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)
