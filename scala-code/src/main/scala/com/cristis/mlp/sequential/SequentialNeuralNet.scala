package com.cristis.mlp.sequential

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.pow
import breeze.stats.distributions.Rand
import com.cristis.mlp.functions.{sigmoid, sigmoidPrime}

/**
  * Class representing a multilayer perceptron with only 1 hidden layer
  * Created by cristian.schuszter on 6/23/2017.
  *
  *
  */
class SequentialNeuralNet(inputLayerSize: Int,
                          outputLayerSize: Int = 1,
                          hiddenLayers: List[Int] = List(3),
                          alpha: Double = 1) {

  if (hiddenLayers == null || hiddenLayers.length < 1) {
    throw new IllegalArgumentException("There must be at least one hidden layer")
  }

  /**
    * The second set of weights, mapping from the hidden layer to the output layer
    */
  var weights: Array[DenseMatrix[Double]] = buildWeightsForHiddenLayerAndOutputs

  private var applications: Array[DenseMatrix[Double]] = new Array[DenseMatrix[Double]](weights.length)
  private var activations: Array[DenseMatrix[Double]] = new Array[DenseMatrix[Double]](weights.length)
  private var yhat: DenseMatrix[Double] = _

  private def buildWeightsForHiddenLayerAndOutputs: Array[DenseMatrix[Double]] = {
    val w = hiddenLayers.indices.map {
      i =>
        if (i == 0) {
          DenseMatrix.rand(inputLayerSize, hiddenLayers.head, Rand.gaussian)
        }
        else {
          DenseMatrix.rand(hiddenLayers(i - 1), hiddenLayers(i), Rand.gaussian)
        }
    }
    (w ++ Seq(DenseMatrix.rand(hiddenLayers.last, outputLayerSize, Rand.gaussian))).toArray
  }

  def isRegressionNet: Boolean = outputLayerSize == 1

  /**
    * Computes the forward propagation based on the current weights. Once the network is trained, this function is used to
    * classify future instances of objects.
    *
    * @param x matrix of inputs that needs forward propagation through the network
    */
  def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    applications(0) = x * weights.head
    activations(0) = applications.head.map(n => sigmoid(n))
    // apply sigmoid activation to each member of the matrix
    for(i <- 1 until weights.length) {
      applications(i) = activations(i-1) * weights(i)
      activations(i) = applications(i).map(n => sigmoid(n))
    }
    activations.last
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
  def backpropagate(x: DenseMatrix[Double], y: DenseMatrix[Double]): List[DenseMatrix[Double]] = {
    yhat = this.forward(x)
    var gradients: Array[DenseMatrix[Double]] = new Array(weights.length)
    var lastDelta: DenseMatrix[Double] = null


    for(i <- applications.length-1 to 0 by -1) {
        if(i == applications.length-1) {
          lastDelta = -(y - yhat) :* applications(i).map(z => sigmoidPrime(z))
          gradients(i) = activations(i-1).t * lastDelta
        } else if(i == 0) {
          lastDelta = (lastDelta * weights(i+1).t) :* applications(i).map(z => sigmoidPrime(z))
          gradients(i) = x.t * lastDelta
        } else {
          lastDelta = (lastDelta * weights(i+1).t) :* applications(i).map(z => sigmoidPrime(z))
          gradients(i) = activations(i-1).t * lastDelta
        }
    }
    gradients.toList
  }

  /**
    * Do one iteration of the neural network, computing backpropagation and apply the new changes to the weight matrices
    *
    * @param x the matrix of inputs
    * @param y the expected values
    */
  def iterate(x: DenseMatrix[Double], y: DenseMatrix[Double]): Unit = {
    val gradients = backpropagate(x, y)
    applyChanges(gradients)
  }


  private def applyChanges(gradients: List[DenseMatrix[Double]]): Unit = {
    gradients.indices.foreach {
      i =>
        weights(i) = weights(i) - gradients(i).map(d => d * alpha)
    }
  }

}
