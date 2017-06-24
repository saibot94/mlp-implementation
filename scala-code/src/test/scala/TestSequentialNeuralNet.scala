import breeze.linalg.{*, DenseMatrix, DenseVector, linspace, max}
import breeze.plot._
import com.cristis.mlp.sequential.SequentialNeuralNet
import org.scalatest.{Matchers, WordSpec}
/**
  * Created by cristian.schuszter on 6/23/2017.
  */
class TestSequentialNeuralNet extends WordSpec with Matchers {
  def train(net: SequentialNeuralNet, inputMat: DenseMatrix[Double], y: DenseMatrix[Double], its: Int = 10000): Array[Double] = {
    var costs = Array[Double]()
    var newCost = 100d
    var oldCost = 100d
    for (i <- 1 to its) {
      net.iterate(inputMat, y)
      newCost = net.costFunction(inputMat, y)
      costs :+= newCost
      oldCost should be >= newCost
      oldCost = newCost
      if(newCost <= 0.0001d || newCost == 0)
        return costs
    }
    costs
  }

  "SequentialNeuralNet.forward" when {
    "forward propagating" should {
      "compute result" in {
        val net = new SequentialNeuralNet(inputLayerSize = 2)
        val inputMat = DenseMatrix((1d, 1d), (0d, 1d), (1d, 0d), (0d, 0d))
        val y = DenseMatrix(0d, 1d, 1d, 0d)
        println(net.forward(inputMat))

        println("Cost fct: ")
        println(net.costFunction(inputMat, y))
      }

      "train the network and lower cost at each step" in {

        val net = new SequentialNeuralNet(inputLayerSize = 2, hiddenLayers= List(3,3), alpha = 1)
        val inputMat = DenseMatrix((1d, 1d), (0d, 1d), (1d, 0d), (0d, 0d))
        val y = DenseMatrix(0d, 1d, 1d, 0d)
        println(net.forward(inputMat))
        val costs = train(net, inputMat, y)
        val predict0 = net.forward(DenseMatrix((1d, 1d)))
        val predict1 = net.forward(DenseMatrix((1d, 0d)))
        predict0.data(0) shouldBe 0d +- 0.1d
        predict1.data(0) shouldBe 1d +- 0.1d

        println("predicted: \n" + net.forward(inputMat))
        println("expected: \n" + y )


        val f = Figure()
        val p = f.subplot(0)
        val yc = DenseVector[Double](costs)
        p.xlabel = "epoch"
        p.ylabel = "cost"
        val x = DenseVector[Double]((0 until yc.length).map(_.toDouble).toArray)
        p += plot(x, yc)

        f.height = 768
        f.width = 1366
        f.saveas("perceptron-xor-costs.png")
      }
      "train the network and lower cost at each step for a more complicated example" in {
        val net = new SequentialNeuralNet(inputLayerSize = 2, hiddenLayers = List(4), alpha = 1)
        val inputMat = DenseMatrix((3d, 5d), (5d, 1d), (10d, 2d))
        val y = DenseMatrix(75d, 82d, 93d)
        val ynorm = y :/ 100d

        val inputMatReg = inputMat(*, ::) / max(inputMat(*, ::))
        val costs = train(net, inputMatReg, ynorm)
        val predict0 = net.forward(inputMatReg)
        predict0.data(0) * 100d shouldBe 75d +- 1d

        println("predicted: \n" + net.forward(inputMatReg))
        println("expected: \n" + ynorm )

        val f = Figure()
        val p = f.subplot(0)
        val yc = DenseVector[Double](costs)
        p.xlabel = "epoch"
        p.ylabel = "cost"
        val x = DenseVector[Double]((0 until yc.length).map(_.toDouble).toArray)
        p += plot(x, yc)

        f.height = 768
        f.width = 1366
        f.saveas("sleep-grades-costs.png")
      }
    }
  }

}
