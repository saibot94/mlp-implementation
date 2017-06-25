package com.cristis.mlp.util

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

/**
  * Created by chris on 6/24/2017.
  */
class IdxImageWriter(rowNr: Int, colNr: Int) {

  def write(fileName: String, imageBytes: Array[Int]): Boolean = {
    try {

      val image = new BufferedImage(colNr, rowNr, BufferedImage.TYPE_INT_ARGB)
      val imagePixels = imageBytes.map { gray =>
        0xFF000000 | (gray << 16) | (gray << 8) | gray
      }
      image.setRGB(0, 0, colNr, rowNr, imagePixels, 0, colNr)
      val outputfile = new File(fileName + ".png")
      ImageIO.write(image, "png", outputfile)
      true
    } catch {
      case e: Exception =>
        e.printStackTrace()
        false
    }
  }

  def getPlotImageData(imageBytes: Array[Int]): Array[Int] = {
    imageBytes.map { gray =>
      0xFF000000 | (gray << 16) | (gray << 8) | gray
    }
  }

}
