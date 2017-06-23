package com.cristis.mlp.sequential

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.pow
import breeze.stats.distributions.Rand
import com.cristis.mlp.functions.{sigmoid, sigmoidPrime}

/**
  * Class representing a multilayer perceptron with only 1 hidden layer
  * Created by cristian.schuszter on 6/23/2017.
  */
class SequentialNeuralNet(inputLayerSize: Int,
                          hiddenLayerSize: Int = 3,
                          outputLayerSize: Int = 1,
                          alpha: Double = 1) {

  /**
    * The first set of weights, mapping from the input to the first hidden layer
    */
  var w1: DenseMatrix[Double] = DenseMatrix.rand(inputLayerSize, hiddenLayerSize, Rand.gaussian)

  /**
    * The second set of weights, mapping from the hidden layer to the output layer
    */
  var w2: DenseMatrix[Double] = DenseMatrix.rand(hiddenLayerSize, outputLayerSize, Rand.gaussian)

  private var z2: DenseMatrix[Double] = _
  private var a2: DenseMatrix[Double] = _
  private var z3: DenseMatrix[Double] = _
  private var yhat: DenseMatrix[Double] = _

  def isRegressionNet: Boolean = outputLayerSize == 1

  /**
    * Computes the forward propagation based on the current weights
    *
    * @param x matrix of inputs that needs forward propagation through the network
    */
  def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    z2 = x * w1
    // apply sigmoid activation to each member of the matrix
    a2 = z2.map(n => sigmoid(n))
    z3 = a2 * w2
    z3.map(n => sigmoid(n))
  }

  def costFunction(x: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {
    yhat = this.forward(x)
    0.5d * sum(pow(y - yhat, 2))
  }

  /**
    * Backpropagate and compute the gradients, the derivatives of J (the cost function) with respect to W1 and W2
    *
    * @param x
    * @param y
    * @return a pair of matrices for the dJW1 and dJW2 gradients
    */
  def backpropagate(x: DenseMatrix[Double], y: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    yhat = this.forward(x)
    val delta3: DenseMatrix[Double] = -(y - yhat) :*  z3.map(z => sigmoidPrime(z))
    val dJdw2 = a2.t * delta3

    val delta2 = (delta3 * w2.t)  :* z2.map(z => sigmoidPrime(z))
    val dJw1 = x.t * delta2
    (dJw1, dJdw2)
  }

  /**
    * Do one iteration of the neural network, computing backpropagation and apply the new changes to the weight matrices
    *
    * @param x the matrix of inputs
    * @param y the expected values
    */
  def iterate(x: DenseMatrix[Double], y: DenseMatrix[Double]): Unit = {
    val (djw1, djw2) = backpropagate(x, y)
    applyChanges(djw1, djw2)
  }


  private def applyChanges(djw1: DenseMatrix[Double], djw2: DenseMatrix[Double]): Unit = {
    w1 = w1 - djw1.map(d => d * alpha)
    w2 = w2 - djw2.map(d => d * alpha)
  }

}
