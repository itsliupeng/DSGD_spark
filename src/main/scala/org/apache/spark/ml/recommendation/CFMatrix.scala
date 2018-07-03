package org.apache.spark.ml.recommendation

import com.github.fommil.netlib.BLAS.{getInstance => blas}

import scala.util.Random

object CFMatrix {
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

  def batchUpdate(WW: Map[Int, Array[Double]], HH: Map[Int, Array[Double]], ratingByW: Map[Int, Seq[(Int, Double)]], ratingByH: Map[Int, Seq[(Int, Double)]], numFactor: Int) = {
    val alpha = 0.1
    val lambda = 0.01
    val coff = 1.0F - lambda * alpha

    val newWW: Map[Int, Array[Double]] = WW.map { case (i, wi) =>
      val rating = ratingByW.getOrElse(i, Seq.empty)
      if (rating.nonEmpty) {
        val H: Array[Double] = rating.flatMap(i => HH(i._1)).toArray
        val V = rating.map(_._2).toArray
        val Hx = numFactor
        val Hy = rating.size
        blas.dgemv("T", Hx, Hy, 1.0F, H, Hx, wi, 1, -1.0F, V, 1)
        val wiCopy = wi.clone()
        blas.dgemv("N", Hx, Hy, -alpha / Hy, H, Hx, V, 1, coff, wiCopy, 1)
        (i, wiCopy)
      } else {
        (i, wi)
      }
    }
    val newHH: Map[Int, Array[Double]] = HH.map { case (j, hj) =>
      val rating = ratingByH.getOrElse(j, Seq.empty)
      if (rating.nonEmpty) {
        val W: Array[Double] = rating.flatMap(i => WW(i._1)).toArray
        val V = rating.map(_._2).toArray
        val Wx = numFactor
        val Wy = rating.size
        blas.dgemv("T", Wx, Wy, 1.0F, W, Wx, hj, 1, -1.0F, V, 1)
        blas.dgemv("N", Wx, Wy, -alpha / Wy, W, Wx, V, 1, coff, hj, 1)
      }
      (j, hj)
    }
    (newWW, newHH)
  }

  def evaluation(rating: Seq[(Int, Int, Double)], W: Array[Array[Double]], H: Array[Array[Double]], numRating: Int) = {
    var error = 0.0D
    for (record <- rating) {
      val (i, j, v) = record
      error += Math.pow(v - dot(W(i), H(j)), 2)
    }
    error / numRating
  }

  def evaluation(rating: Seq[(Int, Int, Double)], W:  Map[Int, Array[Double]], H:  Map[Int, Array[Double]]) = {
    var error = 0.0D
    for (record <- rating) {
      val (i, j, v) = record
      error += Math.pow(v - dot(W(i), H(j)), 2)
    }
    error / rating.size
  }

  def main(args: Array[String]): Unit = {
    val numFactor: Int = 30
    val numIter: Int = 1000
    val lambda_v: Double = 0.1D
    val alpha: Double = 0.7D

    val rating: Seq[(Int, Int, Double)] = scala.io.Source.fromFile("/Users/liupeng/Downloads/ml-1m/ratings.dat").getLines().map(parseLine(_)).toList

    val numRating: Int = rating.size
    val numUser: Int = rating.map(_._1).distinct.size
    val numItem = rating.map(_._2).distinct.size
    val maxUser: Int = rating.map(_._1).max
    val maxItem = rating.map(_._2).max

    val ratingByW: Map[Int, Seq[(Int, Double)]] = rating.groupBy(_._1).map { case (i, s) => (i, s.map { case (_, i, v) => (i, v)}) }
    val ratingByH: Map[Int, Seq[(Int, Double)]] = rating.groupBy(_._2).map { case (j, s) => (j, s.map { case (u, _, v) => (u, v)}) }

    println(s"numRating $numRating, numUser $numUser, numItem $numItem, maxUser $maxUser, maxItem $maxItem, ratingByW ${ratingByW.size} ratingByH ${ratingByH.size}")

    var totalTime = 0.0D
    var WW: Map[Int, Array[Double]] = Array.fill(maxUser + 1)(Array.fill(numFactor)(Random.nextGaussian())).zipWithIndex.map { case (a, i) => (i, a)}.toMap
    var HH: Map[Int, Array[Double]] = Array.fill(maxItem + 1)(Array.fill(numFactor)(Random.nextGaussian())).zipWithIndex.map { case (a, j) => (j, a)}.toMap
    for (i <- 0.until(numIter)) {
      val start = System.currentTimeMillis() / 1000
      val result =  batchUpdate(WW, HH, ratingByW, ratingByH, numFactor)
      WW = result._1
      HH = result._2
      val error = evaluation(rating, WW, HH)
      val timeElpased = System.currentTimeMillis() / 1000 - start
      totalTime += timeElpased
      println(s"loss of iter $i in time ${timeElpased} totalTime ${totalTime} = $error")
    }

    println("Done")
  }
}
