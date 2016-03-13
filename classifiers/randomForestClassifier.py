from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RFClassifier:

	def __init__(self):
		"""
		This is the constructor responsible for initializing the classifier
		"""
		self.outputHeader = "#rf"
		self.clf = None

	def buildModel(self):
		"""
		This builds the model of the Random Forest Classifier
		"""
		self.clf = RandomForestClassifier(n_estimators=5, max_depth=None,
			 random_state=0)

	def trainRF(self,X, Y):
		"""
		Training the Random Forest Classifier
		"""
		self.clf.fit(X, Y)

	def validateRF(self,X, Y):
		"""
		Validate the Random Forest Classifier
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)

	def testRF(self,X, Y):
		"""
		Test the Random Forest Classifier
		"""
		YPred = self.clf.predict(X)
		print accuracy_score(Y, YPred)