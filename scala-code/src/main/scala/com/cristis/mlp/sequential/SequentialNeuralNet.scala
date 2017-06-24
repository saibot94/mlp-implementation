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
                          alpha: Double = 1,
                          lambda: Double = 0.0001) {

  if (hiddenLayers == null || hiddenLayers.length < 1) {
    throw new IllegalArgumentException("There must be at least one hidden layer")
  }

  /**
    * The second set of weights, mapping from the hidden layer to the output layer
    */
  var weights: Array[DenseMatrix[Double]] = buildWeightsForHiddenLayerAndOutputs

  /**
    * The elements of the activation on each layer. They are matrices of prevLayerSize x nextLayerSize
    */
  private var zs: Array[DenseMatrix[Double]] = new Array[DenseMatrix[Double]](weights.length)

  /**
    * The applications of the sigmoid functions on each of the z matrices on each hidden layer
    */
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
    zs(0) = x * weights.head
    activations(0) = zs.head.map(n => sigmoid(n))
    // apply sigmoid activation to each member of the matrix
    for (i <- 1 until weights.length) {
      zs(i) = activations(i - 1) * weights(i)
      activations(i) = zs(i).map(n => sigmoid(n))
    }
    activations.last
  }

  def costFunction(x: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {
    yhat = this.forward(x)
    val sumOfSquareWeights = weights.map(w => pow(sum(w), 2)).toList.sum
    0.5d * sum(pow(y - yhat, 2)) / x.rows + ((this.lambda / 2d) * sumOfSquareWeights)
  }

  /**
    * Backpropagate and compute the gradients, the derivatives of J (the cost function) with respect to W1 and W2
    *
    * @param x the input matrix
    * @param y the labels of the training data
    * @return a list of matrices representing the gradients of the cost function w.r.t. the weight matrices
    *         of each layer
    */
  def backpropagate(x: DenseMatrix[Double], y: DenseMatrix[Double]): List[DenseMatrix[Double]] = {
    yhat = this.forward(x)
    var gradients: Array[DenseMatrix[Double]] = new Array(weights.length)
    var lastDelta: DenseMatrix[Double] = null


    for (i <- zs.length - 1 to 0 by -1) {
      if (i == zs.length - 1) {
        lastDelta = -(y - yhat) :* zs(i).map(z => sigmoidPrime(z))
        val noReg: DenseMatrix[Double] = (activations(i - 1).t * lastDelta).map(r => r / x.rows)
        gradients(i) = noReg + (weights(i) :* this.lambda)
      } else if (i == 0) {
        lastDelta = (lastDelta * weights(i + 1).t) :* zs(i).map(z => sigmoidPrime(z))
        val noReg: DenseMatrix[Double] = (x.t * lastDelta).map(r => r / x.rows)
        gradients(i) = noReg + (weights(i) :* this.lambda)
      } else {
        lastDelta = (lastDelta * weights(i + 1).t) :* zs(i).map(z => sigmoidPrime(z))
        val noReg: DenseMatrix[Double] = (activations(i - 1).t * lastDelta).map(r => r / x.rows)
        gradients(i) = noReg + (weights(i) :* this.lambda)
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

  /**
    * Train this neural network
    *
    * @param trainX the X dataset, essentially the training set
    * @param trainY the labels for the training dataset
    * @param testX  the test dataset
    * @param testY  the labels for the test dataset
    * @param its    the number of max iterations that the algorithm should take
    * @return
    */
  def train(trainX: DenseMatrix[Double],
            trainY: DenseMatrix[Double],
            testX: DenseMatrix[Double],
            testY: DenseMatrix[Double],
            its: Int = 10000,
            debug: Boolean = false): (Array[Double], Array[Double]) = {

    var costs = Array[Double]()
    var testCosts = Array[Double]()
    var newCost = 100d
    var oldCost = 100d
    for (i <- 1 to its) {
      if(debug) {
        println(s"INFO: NN Iteration #$i")
      }
      if (i % 50 == 0){
        println(s"INFO: NN Iteration #$i")
        println(s"Cost is now ${costs.last}")
        println(s"Test cost is now ${testCosts.last}")
      }
      this.iterate(trainX, trainY)
      newCost = this.costFunction(trainX, trainY)
      costs :+= newCost
      testCosts :+= this.costFunction(testX, testY)
      if(debug) {
        println(s"Cost is now ${costs.last}")
        println(s"Test cost is now ${testCosts.last}")
      }
      if (newCost <= 0.0001d || newCost == 0) {
        println(s"INFO: NN ran for ${costs.length} steps")
        return (costs, testCosts)
      }
//      if (testCosts.length > 1 && testCosts.last + 0.5 > testCosts(testCosts.length - 2)) {
//        println("WARN: Stopping early to avoid overfitting!!")
//        println(s"INFO: NN ran for ${costs.length} steps")
//        return (costs, testCosts)
//      }
      oldCost = newCost
    }
    (costs, testCosts)
  }

}
