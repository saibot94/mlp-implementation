import numpy as np
from scipy import optimize


def sigmoid_activation(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return np.exp(-z) / ((1.0 + np.exp(-z))**2)


class Trainer(object):
    def __init__(self, N):
        self.N = N

    def costFctWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.cost_function(X, y)
        grad = self.N.compute_gradients(X, y)

        return cost, grad

    def callback(self, params):
        self.N.setParams(params)
        self.J.append(self.N.cost_function(self.X, self.y))

    def train(self, X, y):
        params0 = self.N.getParams()
        self.X = X
        self.y = y
        options = {'maxiter': 200, 'disp': True}
        self.J = []
        _res = optimize.minimize(self.costFctWrapper,
                                 params0,
                                 jac=True,
                                 method='BFGS', args=(X, y),
                                 options=options,
                                 callback=self.callback
                                 )

        self.N.setParams(_res.x)
        self.results = _res


class NeuralNetwork(object):

    def __init__(self, feature_no,
                 hidden_layer_neurons=3,
                 output_layer_size=1,
                 alpha=1):
        """
        Create a neural network with a few params avaiable.

        Parameters
        ----------

        feature_no - the size of the features (number of columns in the input matrix)

        hidden_layer_neurons - the number of neurons in the hidden layer 

        output_layer_size - number of possible classes (one output in the case of regression)

        alpha - the step for the gradient descent algorithm
        """
        self.feature_no = feature_no
        self.hidden_layer_neurons = hidden_layer_neurons
        self.output_layer_size = output_layer_size
        self.alpha = alpha

        self.w1 = np.random.randn(self.feature_no, self.hidden_layer_neurons)
        self.w2 = np.random.randn(
            self.hidden_layer_neurons, self.output_layer_size)

    def is_regression_network(self):
        return self.output_layer_size == 1

    def forwardPropagation(self, x):
        self.z2 = np.dot(x, self.w1)
        self.a2 = sigmoid_activation(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        response = sigmoid_activation(self.z3)
        return response

    def cost_prime_function(self, x, y):
        self.yhat = self.forwardPropagation(x)

        # error of the first derivation
        delta3 = np.multiply(-(y - self.yhat), sigmoid_derivative(self.z3))
        dJdw2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.w2.T) * sigmoid_derivative(self.z2)
        dJdw1 = np.dot(x.T, delta2)
        return dJdw1, dJdw2

    def apply_changes(self, dJdw1, dJdw2):
        self.w1 = self.w1 - (self.alpha * dJdw1)
        self.w2 = self.w2 - (self.alpha * dJdw2)

    def cost_function(self, x, y):
        """
        Estimate the error of the classifier by using the sum of squared errors sigma(e^2). This is what we'll have to minimize
        """
        self.yhat = self.forwardPropagation(x)
        return 0.5 * sum((y - self.yhat)**2)

    def iterate(self, X, y):
        dJdw1, dJdw2 = self.cost_prime_function(X, y)

        self.apply_changes(dJdw1, dJdw2)

        print("New error is: ")
        print(str(self.cost_function(self.yhat, y)))
        print("Prediction is: ")
        print(str(self.yhat))

    def compute_gradients(self, x, y):
        dJdw1, dJdw2 = self.cost_prime_function(x, y)
        print("dj1: " + str(dJdw1))
        print("dj2: " + str(dJdw2))
        return np.concatenate((dJdw1.ravel(), dJdw2.ravel()))

    # Helper Functions for interacting with other classes:
    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.w1.ravel(), self.w2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hidden_layer_neurons * self.feature_no
        self.w1 = np.reshape(params[W1_start:W1_end],
                             (self.feature_no, self.hidden_layer_neurons))
        W2_end = W1_end + self.hidden_layer_neurons * self.output_layer_size
        self.w2 = np.reshape(
            params[W1_end:W2_end], (self.hidden_layer_neurons, self.output_layer_size))
