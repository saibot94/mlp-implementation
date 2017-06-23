package com.cristis.mlp.sequential

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Rand
import com.cristis.mlp.functions.sigmoid

/**
  * Class representing a multilayer perceptron with only 1 hidden layer
  * Created by cristian.schuszter on 6/23/2017.
  */
class SequentialNeuralNet(inputLayerSize: Int, hiddenLayerSize: Int = 3, outputLayerSize: Int = 1) {

  /**
    * The first set of weights, mapping from the input to the first hidden layer
    */
  val w1: DenseMatrix[Double] = DenseMatrix.rand(inputLayerSize, hiddenLayerSize, Rand.gaussian)

  /**
    * The second set of weights, mapping from the hidden layer to the output layer
    */
  val w2: DenseMatrix[Double] = DenseMatrix.rand(hiddenLayerSize, outputLayerSize, Rand.gaussian)

  private var z2: DenseMatrix[Double] = _
  private var a2: DenseMatrix[Double] = _
  private var z3: DenseMatrix[Double] = _

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

}
