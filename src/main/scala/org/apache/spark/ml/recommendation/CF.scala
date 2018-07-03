package org.apache.spark.ml.recommendation

import scala.util.Random

object CF {
  def parseLine(line: String) = {
    val s = line.split("::")
    (s(0).toInt - 1, s(1).toInt -1, s(2).toDouble)
  }

  @inline
  def dot[T <: Double](u: Array[T], v: Array[T]) = {
    require(u.length == v.length)
    0.until(u.length).map(i => u(i) * v(i)).sum
  }

  @inline
  def ax[T <: Double](a: T, x: Array[T]): Array[Double] = {
    x.map(_ * a)
  }

  @inline
  def axy[T <: Double](a: T, x: Array[T], y: Array[T]): Array[Double] = {
    ax(a, x).zip(y).map { case (a, b) => a + b}
  }

  @inline
  def axby[T <: Double](a: T, x: Array[T], b: T, y: Array[T]): Array[Double] = {
    ax(a, x).zip(ax(b, y)).map { case (a, b) => a + b}
  }

  def updateWeights(rating: Seq[(Int, Int, Double)], W: Array[Array[Double]], H: Array[Array[Double]], beta_v: Double, lamdba_v: Double, numUser: Int, numItem: Int) = {
    var t = 0
    for (record <- rating) {
      val alpha = 0.01
      t += 1
      val (i, j, v) = record
      val tmp: Double = -1 * (v - dot(W(i), H(j)))
      val gradW = axby(tmp, H(j), lamdba_v / numUser, W(i))
      val gradH = axby(tmp, W(i), lamdba_v / numItem, H(j))

      W(i) = axy(-alpha, gradW, W(i))
      H(j) = axy(-alpha, gradH, H(j))
      val z = 0
    }
    (W, H)
  }

  def evaluation(rating: Seq[(Int, Int, Double)], W: Array[Array[Double]], H: Array[Array[Double]], numRating: Int) = {
    var error = 0.0D
    for (record <- rating) {
      val (i, j, v) = record
      error += Math.pow(v - dot(W(i), H(j)), 2)
    }
    error / numRating
  }

  def main(args: Array[String]): Unit = {
    val numFactor: Int = 100
    val numIter: Int = 1000
    val lambda_v: Double = 0.1D
    val alpha: Double = 0.7D

    val rating: Seq[(Int, Int, Double)] = scala.io.Source.fromFile("/Users/liupeng/Downloads/ml-1m/ratings.dat").getLines().map(parseLine(_)).toList

    val numRating: Int = rating.size
    val numUser: Int = rating.map(_._1).distinct.size
    val numItem = rating.map(_._2).distinct.size
    val maxUser: Int = rating.map(_._1).max
    val maxItem = rating.map(_._2).max

    println(s"numRating $numRating, numUser $numUser, numItem $numItem, maxUser $maxUser, maxItem $maxItem")

    val W: Array[Array[Double]] = Array.fill(maxUser + 1)(Array.fill(numFactor)(Random.nextGaussian()))
    val H: Array[Array[Double]] = Array.fill(maxItem + 1)(Array.fill(numFactor)(Random.nextGaussian()))

    var totalTime = 0.0D
    for (i <- 0.until(numIter)) {
      val start = System.currentTimeMillis() / 1000
      updateWeights(rating, W, H, alpha, lambda_v, numUser, numItem)
      val error = evaluation(rating, W, H, numRating)
      val timeElpased = System.currentTimeMillis() / 1000 - start
      totalTime += timeElpased
      totalTime += timeElpased
      println(s"loss of iter $i in time ${timeElpased} totalTime ${totalTime} = $error")
    }

    println("Done")
  }

}
