package com.cristis.mlp
import scala.math.{exp, pow}
/**
  * Created by cristian.schuszter on 6/23/2017.
  */
package object functions {

  def sigmoid(x: Double): Double = 1d / (1d + exp(-x))

  def sigmoidPrime(x: Double): Double = exp(-x) / pow(1d + exp(-x), 2)

}
