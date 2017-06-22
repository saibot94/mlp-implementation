from pymlp.neural_network import NeuralNetwork
import numpy as np

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


if __name__ == '__main__':
    test_forward_prop()
