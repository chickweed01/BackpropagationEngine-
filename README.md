# The Backpropagation Engine
> A stand alone class library for performing backpropagation on a neural network

## Introduction

Backpropagation is the process of adjusting the weight of a connector
that joins two nodes based on the gradient of an error value at the output layer. The goal of this project is to provide
an API for adjusting the network's weights. The user simply has to specify the neural network's layer.

## Basic Steps
It begins with the calculation of the sums at each of the 
final output layer nodes. The gradients of the errors at each of the 
final output layer nodes are then calculated. The output gradients are 
applied to the weights between the final hidden layer and output layer nodes.

* Calculate the sums at the output layer
* Calculate the output gradients by multiplying the derivative of the 
 	 output activations times the output error (defined as target - calculated).
* Calculate the hidden layer gradients by multiplying the derivative of the 
 	 hidden output activation times the corresponding output gradient.
* For each layer (excluding the input layer) adjust the weights of the 
	 inbound connections. Don't forget the bias weights at each layer.
 
