package com.cristis.mlp.util

import breeze.linalg.DenseVector
import breeze.plot.{Figure, plot}

/**
  * Created by chris on 6/24/2017.
  */
object PlotUtil {

  def plotNeuralNetworkOutput(costs: Array[Double],
                              trainCosts: Array[Double],
                              fileName: String): Unit = {
    val f = Figure()
    val p = f.subplot(0)
    val yc = DenseVector[Double](costs)
    val yTestc = DenseVector[Double](trainCosts)
    p.xlabel = "epoch"
    p.ylabel = "cost"
    val x = DenseVector[Double]((0 until yc.length).map(_.toDouble).toArray)
    p += plot(x, yc)

    val xTest = DenseVector[Double]((0 until yTestc.length).map(_.toDouble).toArray)
    p += plot(x, yc)
    p += plot(xTest, yTestc, '.')

    f.height = 1080
    f.width = 1920
    f.saveas(s"$fileName.png")
  }


  def plotAccuracy(accuracy: Array[Double],
                   fileName: String): Unit = {
    val f = Figure()
    val p = f.subplot(0)
    val yc = DenseVector[Double](accuracy)
    p.xlabel = "epoch"
    p.ylabel = "accuracy (in percent)"
    val x = DenseVector[Double]((0 until yc.length).map(_.toDouble).toArray)
    p += plot(x, yc)


    f.height = 1080
    f.width = 1920
    f.saveas(s"$fileName.png")
  }

}
