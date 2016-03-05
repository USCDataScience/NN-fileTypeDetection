### File Type Detection

This repository contains files for running Deep Neural Network that help generate NNModel files for Apache Tika.

## Dependecies

- Pandas
- Numpy
- Theano (for leveraging GPU and building deeper netwoks)


## Training Results

```python
Parameter `pieces` is unused for layer type `Sigmoid`.
Parameter `pieces` is unused for layer type `Sigmoid`.
Initializing neural network with 3 layers, 255 inputs and 2 outputs.
Training on dataset of 222,578 samples with 57,202,546 total size.
Terminating after specified 25 total iterations.
0.999588904464
```


| Mime type     		  | Test Accuracy     | Number of Hidden Layers      
| ------------------------|:------------------|:-----------------------
| application/x-grib      | 92.34%			  |  2
| application/x-grib   	  | 94.33%			  |  4
| application/xhtml  	  | 99.5%			  |  2


Current Run Statistics:
Reported accuracy on application/xhtml : 99.5%

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