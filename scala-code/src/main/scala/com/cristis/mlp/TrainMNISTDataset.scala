package com.cristis.mlp

import breeze.linalg.{DenseMatrix, max}
import com.cristis.mlp.sequential.SequentialNeuralNet
import com.cristis.mlp.util.{IdxReader, PlotUtil}

/**
  * Created by chris on 6/24/2017.
  */
object TrainMNISTDataset {
  def main(args: Array[String]): Unit = {
    println("==== Reading training images (60k)")
    val trainFile = "C:\\Users\\chris\\Desktop\\mnist\\train-images.idx3-ubyte"
    val trainLabelFile = "C:\\Users\\chris\\Desktop\\mnist\\train-labels.idx1-ubyte"
    val trainReader = new IdxReader(trainFile, trainLabelFile, limit = Some(1000))

    println("==== Reading test images (10k)")
    val testFile = "C:\\Users\\chris\\Desktop\\mnist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte"
    val testLabelFile = "C:\\Users\\chris\\Desktop\\mnist\\t10k-labels.idx1-ubyte"
    val testReader = new IdxReader(testFile, testLabelFile, limit = Some(100))


    println("==== Successfully read all images, starting training of network...")
    // The input layer will have 28 x 28 neurons = 784 neurons firing
    val inputLayerSize = trainReader.getCols * trainReader.getRows
    val net = new SequentialNeuralNet(inputLayerSize = inputLayerSize, outputLayerSize = 10, alpha=3.0, hiddenLayers = List(30))
    val trainXImages = trainReader.getImages
    val trainYLabels = trainReader.getLabelsForML
    val testXImages = testReader.getImages
    val testYLabels = testReader.getLabelsForML

    val trainX: DenseMatrix[Double] = IdxReader.imageVectorToMatrix(trainXImages).map(x => x / 255d)
    val trainY: DenseMatrix[Double] = IdxReader.imageVectorToMatrix(trainYLabels)

    val testX: DenseMatrix[Double] = IdxReader.imageVectorToMatrix(testXImages).map(x => x / 255d)
    val testY: DenseMatrix[Double] = IdxReader.imageVectorToMatrix(testYLabels)


    val validateX  = IdxReader.imageVectorToMatrix(testXImages.take(10)).map(x => x / 255d)
    val validateY = IdxReader.imageVectorToMatrix(testYLabels.take(10))

    println(testX)
    println("y == ")
    println(testY)

    println("=== Predicted: ")
    println(net.forward(validateX).map(x => scala.math.round(x)))

    println("=== Expected: ")
    println(validateY.map(x => scala.math.round(x)))
    println(validateY.data.mkString(" "))
    val (costs, testCosts) = net.train(trainX, trainY, testX, testY, its = 2000, debug = false)

    println("=== Predicted: ")
    println(net.forward(validateX).map(x => scala.math.round(x)))

    println("=== Expected: ")
    println(validateY.map(x => scala.math.round(x)))
    PlotUtil.plotNeuralNetworkOutput(costs, testCosts, "mnist_full")
  }
}
