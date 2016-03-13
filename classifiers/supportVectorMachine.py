from sklearn import svm
from sklearn.metrics import accuracy_score

class SupportVectorMachineClassifier:

	def __init__(self):
		"""
		This is the constructor responsible for initializing the classifier
		"""
		self.outputHeader = "#svm"
		self.clf = None

	def buildModel(self):
		"""
		This builds the model of the classifier
		"""
		self.clf =  svm.SVC()

	def trainSVM(self,X, Y):
		"""
		Training the Support Vector Machine
		"""
		self.clf.fit(X, Y)

	def validateSVM(self,X, Y):
		"""
		Validate the neural network
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)

	def testSVM(self,X, Y):
		"""
		Test the neural network
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)


