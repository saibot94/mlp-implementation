package com.cristis.mlp

/**
  * Created by cristian.schuszter on 6/23/2017.
  */
package object functions {

  def sigmoid(x: Double): Double = 1d / (1d + scala.math.exp(-x))

}
