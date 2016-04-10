#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
#

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import sklearn
from warnings import warn
import cPickle

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
		self.json = ""

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

	def pickleClassifier(self):
		"""
		pickles the classifier.
		"""
		with open('dtclassifier.pkl', 'wb') as fid:
			cPickle.dump(self.clf, fid)

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
		tree.export_graphviz(self.clf, out_file='tree.dot') 
		self.treeToJson(self.clf, [str(("byte"+str(i+1))) for i in range(0,256)])

	def node_to_str(self, tree, node_id, criterion, feature_names):
		"""
		This function converts the node to string
		"""
		if not isinstance(criterion, sklearn.tree.tree.six.string_types):
			criterion = "impurity"

		value = tree.value[node_id]
		if tree.n_outputs == 1:
			value = value[0, :]
		jsonValue = ', '.join([str(x) for x in value])

		if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
			return '"id": "%s", "criterion": "%s", "impurity": "%s", "samples": "%s", "value": [%s]' \
					% (node_id, 
					criterion,
					tree.impurity[node_id],
					tree.n_node_samples[node_id],
					jsonValue)
		else:
			if feature_names is not None:
				feature = feature_names[tree.feature[node_id]]
			else:
				feature = tree.feature[node_id]

			if "=" in feature:
				ruleType = "="
				ruleValue = "false"
			else:
				ruleType = "<="
				ruleValue = "%.4f" % tree.threshold[node_id]

			return '"id": "%s", "rule": "%s %s %s", "%s": "%s", "samples": "%s"' \
				 % (node_id, 
					feature,
					ruleType,
					ruleValue,
					criterion,
					tree.impurity[node_id],
					tree.n_node_samples[node_id])

	def recurse(self,tree, node_id, criterion, feature_names, parent=None, depth=0):
		"""
		Recursively go through tree nodes and return information

		output: return json details for each node
		"""
		tabs = "  " * depth
		js = ""
	 
		left_child = tree.children_left[node_id]
		right_child = tree.children_right[node_id]
	 
		js = js + "\n" + \
			tabs + "{\n" + \
			tabs + "  " + self.node_to_str(tree, node_id, criterion,feature_names)
	 
		if left_child != sklearn.tree._tree.TREE_LEAF:
			js = js + ",\n" + \
				 tabs + '  "left": ' + \
				 self.recurse(tree, \
					   left_child, \
					   criterion=criterion, \
					   parent=node_id, \
					   depth=depth + 1,
					   feature_names = feature_names) + ",\n" + \
				 tabs + '  "right": ' + \
				 self.recurse(tree, \
					   right_child, \
					   criterion=criterion, \
					   parent=node_id,
					   depth=depth + 1,
					   feature_names = feature_names)
	 
		js = js + tabs + "\n" + \
			 tabs + "}"
	 
		return js

	def treeToJson(self,decision_tree, feature_names=None):
		"""
		This function handles the tree structure and outputs json file with each tree node.

		output example:
		id": "81", "criterion": "decision_tree.criterion", "impurity": "0.0", "samples": "39", "value": [0.0, 39.0]

		"""
		self.json = ""
		print decision_tree.criterion
		if isinstance(decision_tree, sklearn.tree.tree.Tree):
			self.json = self.json + self.recurse(decision_tree, 0, criterion="impurity",feature_names = feature_names)
		else:
			self.json = self.json + self.recurse(decision_tree.tree_, 0, criterion="decision_tree.criterion", feature_names = feature_names)
	 
		with open("dtjson.json","wb") as datafile:
			datafile.write(self.json)
