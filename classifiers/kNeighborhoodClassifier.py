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