from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNNClassifier:
	
	def __init__(self):
		"""
		This is the constructor for the KNN Classifier
		"""
		self.outputHeader = "#knn"
		self.clf = None

	def buildModel(self):
		"""
		This builds the model of the KNN Classifier
		"""
		self.clf = KNeighborsClassifier(n_neighbors=3)

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