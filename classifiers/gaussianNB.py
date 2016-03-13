from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class GaussianNBClassifier:

	def __init__(self):
		"""
		This is the constructor responsible for initializing the classifier
		"""
		self.outputHeader = "#gnb"
		self.clf = None

	def buildModel(self):
		"""
		This builds the model of the Gaussian NB classifier
		"""
		self.clf =  GaussianNB()

	def trainGaussianNB(self,X, Y):
		"""
		Training the Gaussian NB Classifier
		"""
		self.clf.fit(X, Y)

	def validateGaussianNB(self,X, Y):
		"""
		Validate the Gaussian NB Classifier
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)

	def testGaussianNB(self,X, Y):
		"""
		Test the Gaussian NB Classifier
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)