# File Type Detection

This repository contains files for building classifiers for file type detection. Initially the project was built for developing Deep Neural Networks in order to spit out neural network parameters for Tika to learn.

However, I  have built over other classifiers for performing the same functionality. This repository now supports almost 6 classifiers and preprocessors for creating your on dataset.

I highly encourage contributors to develop their own dataset and try out the different classifiers to update the result.

## Current Status

We are working on scaffolding in the form of a [Flask App](app/) that employs a decision tree to classify files and that uses the built model files from the library.

## Dependencies

- Pandas
- Numpy
- Theano (for leveraging GPU and building deeper netwoks)
- Sklearn

## Classifiers Supported

- Decision Tree
- Neural Network
- Gaussian NB
- SVM
- Random Forest Classifier
- K-Nearest Neighbor Classifier
- Gradient Boosting Classifier

### Neural Network Results

| Mime type     		  | Test Accuracy     | Number of Hidden Layers      
| ------------------------|:------------------|:-----------------------
| application/x-grib      | 92.34%			  |  2
| application/x-grib   	  | 94.33%			  |  4
| application/xhtml  	  | 99.5%			  |  2

### Decision Tree Results

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 99.76%

### Support Vector Machine

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 90.85%

### Gaussian Naive Bayes

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 90.30%

### Random Forest Classifier

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 99.94%

### KNN Classifier

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 99.54%

### Stochastic Gradient Descent

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 98.99

### Gradient Boosting Classifier

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 99.91


## Running the project

Each classifier in the classifier package can be used to train your model. Classifiers follow a simple structure which involves three steps:
- build the model
- train the classifier
- test the classifier

The neural network is special and generates a nnmodel file that can be used with Apache Tika in order to train the NN to work on content based detection and not using Magic Numbers.

## Understanding the input file

It is assumed that the input training files have the following format:
- First 256 columns correspond to the byte frequency companded using any function you like
- The last column is the output column.

Wonderng how to go ahead and create dataset like the one used? The preprocessor contains important constructs that will help generate the dataset needed.
