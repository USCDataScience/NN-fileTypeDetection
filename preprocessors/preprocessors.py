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

import sys
import json
import csv
import os
import logging

class Preprocessor:

	def __init__(self, mimeType, outputFileName, beta=1.5):
		"""
		Initializes the directory parameters for loading the dataset.
		initializes the output array for fingerprint data.
		initializes the mime type under consideration.
		"""
		self.beta = beta
		# this is the output list that keeps track of all the fingerprint data.
		self.output = []
		# This keeps track of the mime type under consideration
		self.analysisSelection = mimeType
		self.outputFileName = outputFileName

	def convertToByteTable(self, filename):
		"""
		Converts the contents of the file to a 256 byte array

		input: filename
		output: byte table consisting of frequency distribution
		"""
		try:
			table = [0] * 256
			data = open(fileName, 'rb')
			buff = data.read(2 ** 20)
			while buff:
				for c in buff:
					table[ord(c)] += 1
				buff = data.read(2 ** 20)
			data.close()
			return table
		except:
			self.logger('Usage: %s <filename>' % os.path.basename(sys.argv[0]))

	def compandBFD(self, table):
		"""
		performs beta companding with beta value default as 1.5

		input: byte frequency table
		output: normalizes the values and compands to return a byte array.
		"""
		table = [x * 1.0 / max(table) for x in table]
		table = [(x ** (1. / self.beta)) for x in table]
		return table

	def fileReader(self,filename, outputToFile = False, logging = True):
		try:
			if type(filename) is list:
				result  = 0
				for files in filename:
					result+=1
					self.computeFingerPrint(files)
					if logging:
						self.logger("processed "+files)
			else:
				self.computeFingerPrint(filename)
				if logging:
					self.logger("processed "+filename)

			if outputToFile:
				self.outputToFile()
		except Exception as e:
			logging.warning("No such file found, %s", e)

	def computeFingerPrint(self, filename):
		"""
		Computes the fingerprint for data.

		input: filename
		output: fingerprint data is appended to output file
		"""

		table  = self.convertToByteTable(filename)
		

		table = self.compandBFD(table)
		if key == self.analysisSelection:
			table.append(1)
		else:
			table.append(0)

		table.append(key)
		self.output.append(table)

	def computeOnlyFingerPrint(self, filename):
		table  = self.convertToByteTable(filename)
		table = self.compandBFD(table)
		return table


	def outputToFile(self):
		"""
		Outputs the file to csv

		output: a csv document with the filename
		"""
		header = [str(("byte"+str(i+1))) for i in range(0,256)]
		header.append("output")
		newOutput = []
		newOutput.append(header)
		newOutput.extend(output)
		with open(self.outputFileName, "wb") as f:
			writer = csv.writer(f)
			writer.writerows(newOutput)
