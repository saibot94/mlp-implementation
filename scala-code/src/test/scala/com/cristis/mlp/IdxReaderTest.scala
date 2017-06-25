package com.cristis.mlp

import com.cristis.mlp.util.{IdxImageWriter, IdxReader, PlotUtil}
import org.scalatest.{Matchers, WordSpec}

/**
  * Created by chris on 6/24/2017.
  */
class IdxReaderTest extends WordSpec with Matchers {
  "IdxReader.read" when {
    "initializing" should {
      "read all MNIST data" in {
        val trainFile = "C:\\Users\\chris\\Desktop\\mnist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte"
        val labelFile = "C:\\Users\\chris\\Desktop\\mnist\\t10k-labels.idx1-ubyte"
        val reader = new IdxReader(trainFile, labelFile)
        reader.getImages.length shouldBe 10000
        reader.getLabels.length shouldBe 10000

        reader.getLabelsForML.head shouldBe List(0,0,0,0,0,0,0,1,0,0).toArray
      }
    }

    "writing the file" should {
      "write an image" in {
        val trainFile = "C:\\Users\\chris\\Desktop\\mnist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte"
        val labelFile = "C:\\Users\\chris\\Desktop\\mnist\\t10k-labels.idx1-ubyte"
        val reader = new IdxReader(trainFile, labelFile, Some(10))

        val images = reader.getImages.take(10)
        val labels = reader.getLabels.take(10)

        val writer = new IdxImageWriter(reader.getRows, reader.getCols)
        (0 until 10).foreach {
          i =>
            writer.write(s"mnist_test_file${i+1}_l" + labels(i).toString, images(i)) shouldBe true
        }


      }
    }

    "writing a composite file" should {
      "write a composite image" in {
        val trainFile = "C:\\Users\\chris\\Desktop\\mnist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte"
        val labelFile = "C:\\Users\\chris\\Desktop\\mnist\\t10k-labels.idx1-ubyte"
        val reader = new IdxReader(trainFile, labelFile, Some(10))

        val images = reader.getImages.take(10)
        val labels = reader.getLabels.take(10)
        val writer = new IdxImageWriter(reader.getRows, reader.getCols)

        val mappedImages = images.map(im => writer.getPlotImageData(im)).toList
        PlotUtil.plotImages(reader.getCols, reader.getRows, mappedImages,labels, "composite_number_plot")
      }
    }
  }

}
