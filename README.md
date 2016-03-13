# File Type Detection

This repository contains files for building classifiers for file type detection. Initially the project was built for developing Deep Neural Networks in order to spit out neural network parameters for Tika to learn.

However, I  have built over other classifiers for performing the same functionality.

# Dependecies

- Pandas
- Numpy
- Theano (for leveraging GPU and building deeper netwoks)


# Classifiers Supported

- Decision Tree
- Neural Network
- Gaussian NB
- SVM
- Random Forest Classifier
- K-Nearest Neighbor Classifier

# Training Results (Neural Network)

```python
Parameter `pieces` is unused for layer type `Sigmoid`.
Parameter `pieces` is unused for layer type `Sigmoid`.
Initializing neural network with 3 layers, 255 inputs and 2 outputs.
Training on dataset of 222,578 samples with 57,202,546 total size.
Terminating after specified 25 total iterations.
0.999588904464
```

## Neural Network Results

| Mime type     		  | Test Accuracy     | Number of Hidden Layers      
| ------------------------|:------------------|:-----------------------
| application/x-grib      | 92.34%			  |  2
| application/x-grib   	  | 94.33%			  |  4
| application/xhtml  	  | 99.5%			  |  2

## Decision Tree Results

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 99.76%

## Support Vector Machine

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 90.85%

## Gaussian Naive Bayes

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 90.30%

## Random Forest Classifier

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 99.94%

## KNN Classifier

| Mime type               | Test Accuracy     
|-------------------------|:------------------
| application/x-grib      | 99.54%



## Running the project

Simply execute:

`python
tester.py
`

This will cause the program to generate a nnmodel file that can be used with Tika.

## Understanding the input file

It is assumed that the input training files have the following format:
- First 256 columns correspond to the byte frequency companded using any function you like
- The last column is the output column.