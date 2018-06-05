# Neural Network in Java
A general purpose implementation of a neural network in Java. Capable of handling multiple layers with a given amount of neurons in each layer, the following network implementation is able to generate a precision of up to 97,74% accuracy classifying the MNIST dataset.

## Weight Visualization
The network is capable of firing backwards, which means output values are specified at the output neurons and the neural network is run backwards. This will generate input values which the network "guessed". The following images were generated after "asking" the MNIST-network how it thinks a zero, a three and a seven look like.

![Zero](https://github.com/danielkleebinder/neural-network/blob/master/imgs/number_0.png?raw=true)
![Three](https://github.com/danielkleebinder/neural-network/blob/master/imgs/number_3.png?raw=true)
![Seven](https://github.com/danielkleebinder/neural-network/blob/master/imgs/number_7.png?raw=true)

These three images are rather clear and show, that the network has little to no troubles classifying these numbers. On the other side of the spectrum are numbers like six, eight and nine.

![Six](https://github.com/danielkleebinder/neural-network/blob/master/imgs/number_6.png?raw=true)
![Eight](https://github.com/danielkleebinder/neural-network/blob/master/imgs/number_8.png?raw=true)
![Nine](https://github.com/danielkleebinder/neural-network/blob/master/imgs/number_9.png?raw=true)

The network is also capable of classifying these numbers, but with a little lower accuracy. It is rather interesting to see how computer brains think numbers look like. This feature is often quite useful for debugging huge networks with thousands and thousands of neurons and millions of synapses.

## Activation Functions
The library supports a lot of different activation functions. For example is almost every activation function from the corresponding [Wikipedia article](https://en.wikipedia.org/wiki/Activation_function) implemented. Some of the most common activation functions are:

* Sigmoid
* Softmax
* Linear Rectifiers such as ReLU, SiLU, ELU, etc.
* Maxout
* Identity (or Linear)
* Hyperbolic Tangent (or TanH)
* Binary Step
* Bent Identity
* Gaussian
* Soft Plus
* and many more

A simple interface supports adding new and even more complex activation functions as well.


## Loss Functions
The neural network library supports out of the box many different loss functions. From the easiest and most commonly used one, the quadratic loss function to advanced algorithms like cross-entropy and hellinger. The softmax activation function for example relies heavily on the structure of the cross-entropy loss function.

## Regularization
The regularization method is used to prevent the network from over-fitting considering a specific set of train data. Some neural networks may need regularization methods, especially the ones with very complex learning patterns. Other networks on the other hand (like most XOR-networks) do not need regularization at all.

The following methods are supported out of the box

* Dropout - For example random 20% of all neurons will be left out during learning process.
* L1 - The "Lasse Regression" methods computes a linear offset for each synaptic weight.
* L2 - This squared ridge regression method is very common in neural networks. It depends on the synaptic weights.
* None - Does not regulate the weights at all.

## Other Features
The neural network library also supports many other useful features like

* Builder-classes for the networks, layers, neurons and learning methods
* Different initialization methods like xavier and random
* Normalization methods like Gaussian- and Min-Max-Input-Normalization
* Neuron connection types like randomly or densely connected layers
* etc.

## Example

Simple two hidden layer configuration of a feed forward network. Using the Swish (or [SiLU](https://arxiv.org/pdf/1702.03118.pdf)) activation function and an exponential rectifier.
```Java
NeuralNetwork neuralNetwork = new NeuralNetworkBuilder()
	.layer("Input Layer", 784, new Identity())
	.layer("Hidden layer (Swish)", 90, new SwishRectifier())
	.layer("Hidden layer (ELU)", 45, new ExponentialRectifier())
	.layer("Output Layer", 10, new Softmax())
	.connector(new DenseConnector())
	.initializer(new XavierInitializer())
	.normalization(new MinMax())
	.build();
```
The network uses min-max normalization for input neurons (not really necessary in the MNIST-example), densely connected layers, xavier-weight-initialization and the softmax activation function for the output layer.
```Java
BackPropagation backPropagation = new BackPropagationBuilder()
	.regularization(new Dropout())
	.lossFunction(new CrossEntropy())
	.learningRate(0.2)
	.momentum(0.9)
	.neuralNetwork(neuralNetwork)
	.build();
```
A back propagation algorithm was used for learning. The following example show how a single set of input values is used to train the configured neural network.
```Java
// Set neural network input parameters
neuralNetwork.input(/* MNIST Input Data [784-Dimensions] */);

// Set desired output values
backPropagation.getDesiredOutputValues().clear();
backPropagation.getDesiredOutputValues().addAll(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
backPropagation.getDesiredOutputValues().set(mnistImage.getLabel(), 1.0);

// Learn the network
backPropagation.learn();
```