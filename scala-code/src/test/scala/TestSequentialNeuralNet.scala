import breeze.linalg.DenseMatrix
import com.cristis.mlp.sequential.SequentialNeuralNet
import org.scalatest.{Matchers, WordSpec}

/**
  * Created by cristian.schuszter on 6/23/2017.
  */
class TestSequentialNeuralNet extends WordSpec with Matchers {
  "SequentialNeuralNet.forward" when {
    "forward propagating" should {
      "compute result" in {
        val net = new SequentialNeuralNet(inputLayerSize = 2)
        val inputMat = DenseMatrix((1.0, 3.0), (4.0, 5.0))
        println(net.forward(inputMat))
      }
    }
  }

}
