package com.cristis.mlp.distributed

/**
  * Created by cristian.schuszter on 6/23/2017.
  */

import breeze.linalg.DenseMatrix
import com.cristis.mlp.functions.{sigmoid, sigmoidPrime}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD

class DistributedNeuralNet(inputLayerSize: Int,
                           outputLayerSize: Int = 1,
                           hiddenLayers: List[Int] = List(3),
                           alpha: Double = 1,
                           lambda: Double = 0.0001)
                          (implicit sc: SparkContext){

  if (hiddenLayers == null || hiddenLayers.length < 1) {
    throw new IllegalArgumentException("There must be at least one hidden layer")
  }

  /**
    * The second set of weights, mapping from the hidden layer to the output layer
    */
  //var weights: Array[RowMatrix] = buildWeightsForHiddenLayerAndOutputs

  /**
    * The elements of the activation on each layer. They are matrices of prevLayerSize x nextLayerSize
    */
  //private var zs: Array[DenseMatrix[Double]] = new Array[DenseMatrix[Double]](weights.length)

}
