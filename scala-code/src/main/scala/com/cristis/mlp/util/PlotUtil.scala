package com.cristis.mlp.util

import breeze.linalg.{DenseMatrix, DenseVector}
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


  def plotImages(cols: Int, rows:Int, images: List[Array[Int]], labels: Vector[Int],
                   fileName: String, predicted: Option[List[Int]] = None): Unit = {
    val f = Figure()
    import breeze.plot.image
    var j = 0
    images.foreach {
      im =>
        val pl = DenseMatrix.zeros[Double](rows, cols)
        for(i <- 0 until rows) {
          pl(i,::) := DenseVector(im.reverse.slice(i * rows, i * rows + cols).map(_.toDouble)).t
        }
        f.subplot(5, 5, j) += image(pl.t)
        f.subplot(5, 5, j).title = s"Label: ${labels(j)}"
        if(predicted.isDefined) {
          f.subplot(5, 5, j).title += s"/ Predicted: ${predicted.get(j)} "
        }
        j += 1
    }
    f.height = 1080
    f.width = 1920
    f.saveas(s"$fileName.png")
  }
}
