package com.cristis.mlp

import breeze.linalg.{*, DenseMatrix, DenseVector, max}
import breeze.plot._
import com.cristis.mlp.sequential.SequentialNeuralNet
import org.scalatest.{Matchers, WordSpec}

/**
  * Created by cristian.schuszter on 6/23/2017.
  */
class TestSequentialNeuralNet extends WordSpec with Matchers {


  "SequentialNeuralNet.forward" when {
    "forward propagating" should {
      "compute result" in {
        println("===== perceptron test")
        val net = new SequentialNeuralNet(inputLayerSize = 2)
        val trainX = DenseMatrix((1d, 1d), (0d, 1d), (1d, 0d), (0d, 0d))
        val y = DenseMatrix(0d, 1d, 1d, 0d)
        println(net.forward(trainX))


        println("Cost fct: ")
        println(net.costFunction(trainX, y))
      }

      "train the network and lower cost at each step" in {

        println("===== XOR test")
        val net = new SequentialNeuralNet(inputLayerSize = 2, hiddenLayers = List(3), alpha = 3.0)
        val trainX = DenseMatrix((1d, 1d), (0d, 1d), (1d, 0d), (0d, 0d))
        val y = DenseMatrix(0d, 1d, 1d, 0d)

        val (costs, testCosts) = net.train(trainX, y, trainX, y, its = 10000)

        println("predicted: \n" + net.forward(trainX))
        println("expected: \n" + y)

        val predict0 = net.forward(DenseMatrix((1d, 1d)))
        val predict1 = net.forward(DenseMatrix((1d, 0d)))

        predict0.data(0) shouldBe 0d +- 0.2d
        predict1.data(0) shouldBe 1d +- 0.2d


        val f = Figure()
        val p = f.subplot(0)
        val yc = DenseVector[Double](costs)
        p.xlabel = "epoch"
        p.ylabel = "cost"
        val x = DenseVector[Double]((0 until yc.length).map(_.toDouble).toArray)
        p += plot(x, yc)

        f.height = 1080
        f.width = 1920
        f.saveas("perceptron-xor-costs.png")
      }
      "train the network and lower cost at each step for a more complicated example" in {
        println("===== sleep-grades test")
        val net = new SequentialNeuralNet(inputLayerSize = 2, hiddenLayers = List(4), alpha = 1)
        val trainX = DenseMatrix((3d, 5d), (5d, 1d), (10d, 2d))
        val y = DenseMatrix(75d, 82d, 93d)
        val ynorm = y :/ 100d

        val trainXReg = trainX(*, ::) / max(trainX(*, ::))
        val (costs, testCosts) = net.train(trainXReg, ynorm, trainXReg, ynorm)
        val predict0 = net.forward(trainXReg)
        predict0.data(0) * 100d shouldBe 75d +- 1d


        println("predicted: \n" + net.forward(trainXReg))
        println("expected: \n" + ynorm)

        val f = Figure()
        val p = f.subplot(0)
        val yc = DenseVector[Double](costs)
        p.xlabel = "epoch"
        p.ylabel = "cost"
        val x = DenseVector[Double]((0 until yc.length).map(_.toDouble).toArray)
        p += plot(x, yc)

        f.height = 1080
        f.width = 1920
        f.saveas("sleep-grades-costs.png")
      }
    }
  }

  "Sequential net.train" when {
    "training a network" should {
      "train it without overfitting" in {
        println("===== overfitting test")
        val net = new SequentialNeuralNet(inputLayerSize = 2,
          hiddenLayers = List(3),
          alpha = 1)

        var trainX = DenseMatrix((3d, 5d), (5d, 1d), (10d, 2d), (6d, 1.5d))
        var trainY = DenseMatrix(75d, 82d, 93d, 70d)
        trainY = trainY :/ 100d
        trainX = trainX(*, ::) / max(trainX(*, ::))

        var testX = DenseMatrix((4d, 5.5d), (4.5d, 1d), (9.2d, 2.5d), (6d, 2d))
        var testY = DenseMatrix(70d, 89d, 85d, 75d)
        testY = testY :/ 100d
        testX = testX(*, ::) / max(testX(*, ::))

        val (costs, trainCosts) = net.train(trainX, trainY, testX, testY, its =  5000)


        println("predicted: \n" + net.forward(testX))
        println("expected: \n" + testY)

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
        f.saveas("overfitting.png")

      }
    }
  }

}
