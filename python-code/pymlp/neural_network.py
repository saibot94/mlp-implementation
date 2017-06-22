import numpy as np


def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))


class NeuralNetwork(object):

    def __init__(self, feature_no,
                 hidden_layer_neurons=3,
                 output_layer_size=1,
                 activation_func=sigmoid_activation):
        """
        Create a neural network with a few params avaiable.

        Parameters
        ----------

        feature_no - the size of the features (number of columns in the input matrix)

        hidden_layer_neurons - the number of neurons in the hidden layer 

        output_layer_size - number of possible classes (one output in the case of regression)

        activation_func - the function which takes a matrix and activates each of the outputs
        """
        self.feature_no = feature_no
        self.hidden_layer_neurons = hidden_layer_neurons
        self.output_layer_size = output_layer_size
        self.activation_func = activation_func

        self.w1 = np.random.randn(self.feature_no, self.hidden_layer_neurons)
        self.w2 = np.random.randn(self.hidden_layer_neurons, self.output_layer_size)

    def is_regression_network(self):
        return self.output_layer_size == 1

    def forwardPropagation(self, x):
        self.z2 = np.dot(x, self.w1)
        self.a2 = self.activation_func(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        response = self.activation_func(self.z3)
        return response
