### File Type Detection

This repository contains files for running Deep Neural Network that help generate NNModel files for Apache Tika.

## Dependecies

- Pandas
- Numpy
- Theano (for leveraging GPU and building deeper netwoks)

## Running the project

Simpy execute:

`python
python neuralNetworkTrainer.py
`

This will cause the program to generate a nnmodel file that can be used with Tika.

## Current Performance Statistics

The current program reports an accuracy of 92.34% using 2 hidden layers and has an estimated accuracy of 94.33% over 4 hidden layers.

## Understanding the input file

It is assumed that the input training files have the following format:
- First 256 columns correspond to the byte frequency companded using any function you like
- The last column is the output column.