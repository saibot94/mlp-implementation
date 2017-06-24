package com.cristis.mlp.util

import java.io.FileInputStream

import breeze.linalg.DenseMatrix

/**
  * Created by chris on 6/24/2017.
  * This file reads the IDX3 format in which the dataset for MNIST is found
  */
class IdxReader(imagesFile: String, labelFile: String, limit: Option[Int] = None) {

  private var images: Vector[Array[Int]] = Vector()
  private var labels: Vector[Int] = Vector()
  private var numberOfRows: Int = _
  private var numberOfColumns: Int = _

  def getImages: Vector[Array[Int]] = images

  def getLabels: Vector[Int] = labels

  /**
    * Gets the neuron form of the output
    *
    * Example: 7 returns (0,0,0,0,0,0,0,1,0,0)
    * @return an list of 10 byte arrays
    */
  def getLabelsForML: Vector[Array[Int]] = labels.map {
    l =>
      val trainingLabel = new Array[Int](10)
      trainingLabel(l) = 1
      trainingLabel
  }

  def getRows: Int = numberOfRows

  def getCols: Int = numberOfColumns

  private def readImages(): Unit = {
    val inImage = new FileInputStream(imagesFile)
    val inLabel = new FileInputStream(labelFile)

    val magicNumberImages = (inImage.read << 24) | (inImage.read << 16) | (inImage.read << 8) | inImage.read
    val numberOfImages = (inImage.read << 24) | (inImage.read << 16) | (inImage.read << 8) | inImage.read
    numberOfRows = (inImage.read << 24) | (inImage.read << 16) | (inImage.read << 8) | inImage.read
    numberOfColumns = (inImage.read << 24) | (inImage.read << 16) | (inImage.read << 8) | inImage.read
    val magicNumberLabels = (inLabel.read << 24) | (inLabel.read << 16) | (inLabel.read << 8) | inLabel.read
    val numberOfLabels = (inLabel.read << 24) | (inLabel.read << 16) | (inLabel.read << 8) | inLabel.read
    val numberOfPixels = numberOfRows * numberOfColumns
    (0 until numberOfImages).foreach(i => {
      val imgPixels = new Array[Int](numberOfPixels)
      if (i % 1000 == 0) {
        println(s"IDXREADER: Number of images extracted: $i")
      }
      (0 until numberOfPixels).foreach { p =>
        val gray = 255 - inImage.read
        imgPixels(p) = gray
      }
      images :+= imgPixels
      labels :+= inLabel.read
      if(limit.isDefined) {
        if(i == limit.get){
          return
        }
      }
    })
  }

  readImages()

}


object IdxReader {
  def imageVectorToMatrix(x: Vector[Array[Int]]): DenseMatrix[Double] = {
    DenseMatrix(x.map(a => a.map(d => d.toDouble)): _*)
  }
}