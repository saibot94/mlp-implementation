package com.cristis.mlp

import breeze.linalg.DenseMatrix
import com.cristis.mlp.sequential.SequentialNeuralNet
import com.cristis.mlp.util.PlotUtil

/**
  * Created by chris on 6/25/2017.
  */
object TrainXOR {
  def main(args: Array[String]): Unit = {
    println("===== XOR test")
    val net = new SequentialNeuralNet(inputLayerSize = 2, hiddenLayers = List( 3), alpha = 10)
    val trainX = DenseMatrix((1d, 1d), (0d, 1d), (1d, 0d), (0d, 0d))
    val y = DenseMatrix(0d, 1d, 1d, 0d)

    val testX = DenseMatrix((1d, 1d), (1d, 0d), (0d, 0d), (0d, 1d))
    val testY = DenseMatrix(0d, 1d, 0d, 1d)

    val (costs, testCosts) = net.train(trainX, y, testX, testY, its = 1000)

    println("predicted: \n" + net.forward(trainX))
    println("expected: \n" + y)
    PlotUtil.plotNeuralNetworkOutput(costs, testCosts, "xor_full")
  }
}
