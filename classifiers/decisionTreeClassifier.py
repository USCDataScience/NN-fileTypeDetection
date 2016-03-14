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

		self.criterion = "gini"
		self.splitter = "best"
		self.min_samples_split = 2
		self.max_depth = None

	def setCritertion(self, param):
		"""
		Sets the criterion for decision tree classifier.
		"""
		if param in ["gini","entropy"]:
			self.criterion = param
		else:
			print "failed to update, defaulting to gini."

	def setSplitter(self, param):
		"""
		Sets the splits for decision tree classifier.
		"""
		if param in ["best","random"]:
			self.splitter = param
		else:
			print "failed to update, defaulting to best."

	def setMinSamplesSplit(self, param):
		"""
		sets the min samples for the decision tree classifier
		"""
		self.min_samples_split = param

	def setMaxDepth(self, param):
		"""
		Sets the max depth of the decision tree classifier
		"""
		self.max_depth = param

	def buildModel(self):
		"""
		This builds the model of the classifier
		"""
		self.clf = tree.DecisionTreeClassifier(criterion = self.criterion,
				 splitter = self.splitter, min_samples_split = self.min_samples_split,
				 		max_depth = self.max_depth)

	def trainTree(self,X, Y):
		"""
		Training the Decision Tree Classifier
		"""

		self.clf.fit(X, Y)

	def validateTree(self,X, Y):
		"""
		Validate the Decision Tree Classifier
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)

	def testTree(self,X, Y):
		"""
		Test the Decision Tree Classifier
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)
