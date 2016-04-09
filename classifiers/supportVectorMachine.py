from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import grid_search

class SupportVectorMachineClassifier:

	def __init__(self):
		"""
		This is the constructor responsible for initializing the classifier
		"""
		self.outputHeader = "#svm"
		self.clf = None
		self.kernel = "rbf"
		self.C = None
		self.tol = 1e-3

	def setKernel(self, param):
		"""
		This function sets the kernel parameter of the Support Vector Machine
		"""
		if param in ["rbf","linear","poly","sigmoid"]:
			self.kernel = param
		else:
			print "error in parameter, defaulting to rbf kernel"

	def setTol(self, param):
		"""
		This function is used to set the tolerance value of the SVM
		"""
		if type(param) is float:
			self.tol = param
		else:
			print "there was an error in the parameter value, defaulting to 0.001"

	def setC(self, param):
		"""
		This function sets the C value for the SVM Classifier
		"""
		self.C = param

	def tuneParameter(self, parameters):
		"""
		This function is used for tuning the parameter of SVM.

		example:
		parameters = {'kernel':('linear', 'poly','rbf'), 'C':[1, 10]}
		"""
		self.clf = grid_search.GridSearchCV(svm.SVC(), parameters)

	def buildModel(self):
		"""
		This builds the model of the classifier
		"""
		self.clf =  svm.SVC()

	def trainSVM(self,X, Y):
		"""
		Training the Support Vector Machine
		"""
		print self.clf.fit(X, Y)

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