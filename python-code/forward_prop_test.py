from pymlp.neural_network import NeuralNetwork, Trainer
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def test_forward_prop():
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

    nn = NeuralNetwork(X.shape[1])
    result = nn.forwardPropagation(X)
    assert result.size == y.size

    print("Actual: ")
    print(str(y))
    print("Predicted: ")
    print(str(result))


def test_error_func():
    y = np.array(([1, 2, 3, 4]), dtype=float)
    yhat = np.array(([1, 1, 2, 4]), dtype=float)

    nn = NeuralNetwork(y.shape[0])


def test_gradient_desc_numerical():
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

    X = X / np.amax(X, axis=0)
    y = y / 100.0

    nn = NeuralNetwork(X.shape[1])

    def f(x):
        return x**2

    epsilon = 1e-4
    x = 1.5
    numericGradient = compute_numerical_gradients(nn, X, y)
    nngradients = nn.compute_gradients(X, y)

    print("Grads (normal): ")
    print(str(nngradients))

    print("Numerical gradients:")
    print(str(numericGradient))

    assert (norm(nngradients - numericGradient) /
            norm(nngradients + numericGradient)) < 0.00000001


def test_opt():
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

    X = X / np.amax(X, axis=0)
    y = y / 100.0

    nn = NeuralNetwork(X.shape[1])
    t = Trainer(nn)
    t.train(X, y)
    plt.plot(t.J)
    plt.grid(1)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.savefig('cost-reduction.png')

    print("X prediction: ")
    print(nn.forwardPropagation(X))

    print("y: ")
    print(y)


def test_backprop():
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

    nn = NeuralNetwork(X.shape[1])

    X = X / np.amax(X, axis=0)
    y = y / 100.0
    costs = []
    for i in range(0, 10000):
        dJdw1, dJdw2 = nn.cost_prime_function(X, y)
        nn.apply_changes(dJdw1, dJdw2)
        costs.append(nn.cost_function(X, y))

    print("Predict: ")
    print(nn.forwardPropagation(X))
    print("y: ")
    print(y)

    plt.plot(costs)
    plt.grid(1)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.savefig('cost-reduction-mine.png')


def compute_numerical_gradients(nn, x, y):
    params = nn.getParams()
    nrgrad = np.zeros(params.shape)
    perturb = np.zeros(params.shape)
    e = 1e-4
    for p in range(len(params)):
        perturb[p] = e
        nn.setParams(params + perturb)
        loss2 = nn.cost_function(x, y)

        nn.setParams(params - perturb)
        loss1 = nn.cost_function(x, y)
        nrgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    nn.setParams(params)

    return nrgrad


if __name__ == '__main__':
    test_opt()
    print("==== My backprop test: ")
    test_backprop()
