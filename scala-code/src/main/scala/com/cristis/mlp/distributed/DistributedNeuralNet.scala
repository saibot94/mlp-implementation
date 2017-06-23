package com.cristis.mlp.distributed

/**
  * Created by cristian.schuszter on 6/23/2017.
  */

import com.cristis.mlp.functions.{sigmoid, sigmoidPrime}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class DistributedNeuralNet(inputLayerSize: Int,
                           hiddenLayerSize: Int = 3,
                           outputLayerSize: Int = 1,
                           alpha: Double = 1)
                          (implicit sc: SparkContext){


}
