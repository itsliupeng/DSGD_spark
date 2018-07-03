package org.apache.spark.ml.recommendation

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.DoubleAccumulator
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.{Dependency, ShuffleDependency, SparkConf, SparkContext}

import scala.collection.mutable
import scala.util.hashing.byteswap64

object DSGD {
  type Score = Float
  def parseLine(line: String): (Int, Int, Score) = {
    val s = line.split(",")
    (s(0).toInt - 1, s(1).toInt -1, s(2).toFloat)
  }

  type RT = (Int, Int, Score)

  class KeyPartitioner(partitions: Int) extends org.apache.spark.Partitioner {
    require(partitions >= 0, s"Number of partitions ($partitions) cannot be negative.")

    def numPartitions: Int = partitions


    override def getPartition(key: Any): Int = key match {
      case null => 0
      case a: Int => a % partitions
      case (a: Int, _: Product) => a % partitions
      case (a: Int, _: Any) => a % partitions
      case _ => key.hashCode() % partitions
    }

    override def equals(other: Any): Boolean = other match {
      case h: KeyPartitioner =>
        h.numPartitions == numPartitions
      case _ =>
        false
    }

    override def hashCode: Int = numPartitions
  }

  @inline
  def dot[T <: Score](u: Array[T], v: Array[T]): Score = {
    require(u.length == v.length)
    u.indices.map(i => u(i) * v(i)).sum
  }

  @inline
  def ax[T <: Score](a: T, x: Array[T]): Array[Score] = {
    x.map(_ * a)
  }

  @inline
  def axy[T <: Score](a: T, x: Array[T], y: Array[T]): Array[Score] = {
    ax(a, x).zip(y).map { case (i, j) => i + j}
  }

  @inline
  def axby[T <: Score](a: T, x: Array[T], b: T, y: Array[T]): Array[Score] = {
    ax(a, x).zip(ax(b, y)).map { case (i, j) => i + j}
  }

  def updateWeights(rating: Seq[(Int, Int, Score)], numFactor: Int, W: mutable.Map[Int, Array[Score]], HTuple: (Int, mutable.Map[Int, Array[Score]]), numUser: Long, numItem: Long, alpha: Score, lambdaV: Score, lossAccum: Option[DoubleAccumulator] = None): (mutable.Map[Int, Array[Score]], (Int, mutable.Map[Int, Array[Score]])) = {
    val HBlockId = HTuple._1
    val H = HTuple._2
    for (record <- rating) {
      val (i, j, v) = record
      val tmp: Score = -2 * (v - blas.sdot(numFactor, W(i), 1, H(j), 1))
      val gradW = axby(tmp, H(j), 2 * lambdaV / numUser, W(i))
      val gradH = axby(tmp, W(i), 2 * lambdaV / numItem, H(j))

//      W(i) = axy(-alpha, gradW, W(i))
//      H(j) = axy(-alpha, gradH, H(j))
      blas.saxpy(numFactor,-alpha, gradW, 1, W(i), 1)
      blas.saxpy(numFactor,-alpha, gradH, 1, H(j), 1)
    }

    lossAccum.foreach(accum => {
      var count = 0
      var error = 0.0D
      for (record <- rating) {
        val (i, j, v) = record
        error += Math.pow(v - blas.sdot(numFactor, W(i), 1, H(j), 1), 2)
        count += 1
      }
      accum.add(error / count)
    })

    (W, (HBlockId, H))
  }


  implicit class ToMutableMap[K, V](s: Seq[(K, V)]) {
    def toMutableMap: scala.collection.mutable.Map[K, V] = {
      val b = scala.collection.mutable.Map.newBuilder[K, V]
      s.map { case (k, v) =>
        b.+=((k, v))
      }
      b.result()
    }
  }

  implicit class ToMean(s: Seq[Double]) {
    def toMean: Double = {
     s.sum / s.size
    }
  }

  def evaluation(rating: Seq[(Int, Int, Score)], W: collection.Map[Int, Array[Score]], H: collection.Map[Int, Array[Score]]): Double = {
    var error = 0.0D
    var count = 0
    for (record <- rating) {
      val (i, j, v) = record
      error += Math.pow(v - dot(W(i), H(j)), 2)
      count += 1
    }
    error / count
  }

  def cleanShuffleDependencies[T](deps: Seq[Dependency[_]], blocking: Boolean = false)(implicit sc: SparkContext): Unit = {
    // If there is no reference tracking we skip clean up.
    sc.cleaner.foreach { cleaner =>
      /**
        * Clean the shuffles & all of its parents.
        */
      def cleanEagerly(dep: Dependency[_]): Unit = {
        if (dep.isInstanceOf[ShuffleDependency[_, _, _]]) {
          val shuffleId = dep.asInstanceOf[ShuffleDependency[_, _, _]].shuffleId
          cleaner.doCleanupShuffle(shuffleId, blocking)
        }
        val rdd = dep.rdd
        val rddDeps = rdd.dependencies
        if (rdd.getStorageLevel == StorageLevel.NONE && rddDeps != null) {
          rddDeps.foreach(cleanEagerly)
        }
      }
      deps.foreach(cleanEagerly)
    }
  }

  def deleteCheckFile(checkFile: String): Unit = {
    Option(checkFile).map { file =>
      val path = new Path(file)
      val fs = path.getFileSystem(new Configuration())
      if (fs.exists(path)) {
        fs.delete(path, true)
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(s"mag-${this.getClass.getName}")
      .set("spark.sql.parquet.compression.codec", "snappy")
    val spark = SparkSession.builder.config(conf).getOrCreate()
    implicit val sc = spark.sparkContext
    sc.setLogLevel("WARN")
    val checkFile = s"hdfs://c3prc-hadoop/user/h_user_profile/liupeng11/checkpoint/DSGD_${System.currentTimeMillis()}"
    sc.setCheckpointDir(checkFile)

    val numPartition = if (args.length > 0) args(0).toInt else 3
    val numFactor = if (args.length > 1) args(1).toInt else 10
    val numIter = if (args.length > 2) args(2).toInt else 100
    val ratingsPath = if (args.length > 3) args(3) else "/Users/liupeng/Downloads/ml-1m/ratings.dat"
    val lambda_v = if (args.length > 4) args(4).toFloat else 0.1F
    val alpha = if (args.length > 5) args(5).toFloat else 0.01F
    val isEvaluate = if (args.length > 6) args(6).toBoolean else true
    val checkPointInterval = if (args.length > 7) args(7).toInt else 3

    val numFactorBr = sc.broadcast(numFactor)
    val unifiedPartitioner = new KeyPartitioner(numPartition)

    val seedBr = sc.broadcast(0)

    val ratings: RDD[(Int, Int, Score)] = sc.textFile(ratingsPath, numPartition).map(parseLine).coalesce(numPartition).cache()
    val numRating = ratings.count()
    val numUser = ratings.map(_._1).distinct().count()
    val numItem = ratings.map(_._2).distinct().count()
    println(s"numRating $numRating, numUser $numUser, numItem $numItem, numFactor $numFactor, numPartition $numPartition")

    val itemsBlock: RDD[(Int, (Int, mutable.Map[Int, Array[Score]]))] = ratings.map(_._2).distinct()
      .map(i => (unifiedPartitioner.getPartition(i), i))
      .groupByKey()
      .mapPartitionsWithIndex({ case (pid, iter) =>
        val random = new XORShiftRandom(byteswap64(seedBr.value ^ pid))
        iter.map { case (itemBlockId, s) =>
          val itemMap: mutable.Map[Int, Array[Score]] = s.toSeq.distinct.map(i => (i, Array.fill(numFactorBr.value)(random.nextGaussian().toFloat))).toMutableMap
          (itemBlockId, (itemBlockId, itemMap))
        }
      }, preservesPartitioning = true)
      .partitionBy(unifiedPartitioner)

    val itemsSetBr: Broadcast[Array[(Int, collection.Set[Int])]] = sc.broadcast(itemsBlock.map {case (_, (id, m)) => (id, m.keySet)}.collect())

    val userRatingsBlock: RDD[(Int, (mutable.Map[Int, Array[Score]], Map[Int, Map[Int, Iterable[(Int, Score)]]], Map[Int, Map[Int, Iterable[(Int, Score)]]]))] = ratings.map { case (u, i, v) => (unifiedPartitioner.getPartition(u), (u, i, v)) }
      .groupByKey()
      .mapPartitionsWithIndex { case (pid, iter) =>
        val random = new XORShiftRandom(byteswap64(seedBr.value ^ pid))
        val itemsSet = itemsSetBr.value
        iter.map { case (userBlockId, ss) =>
          val userSet = ss.map(_._1).toSeq.distinct
          val userMap: mutable.Map[Int, Array[Score]] = userSet.map(u => (u, Array.fill(numFactorBr.value)(random.nextGaussian().toFloat))).toMutableMap
          val subRating: Map[Int, Iterable[(Int, Int, Score)]] = itemsSet.map { case (itemBlockId, f) => (itemBlockId, ss.filter(i => f.contains(i._2)))}.toMap
          val subRatingByUser = subRating.map { case (itemBlockId, rs) => (itemBlockId, rs.groupBy(_._1).map { case (userId, s) => (userId, s.map { case (_, i, r) => (i, r)})})}
          val subRatingByItem = subRating.map { case (itemBlockId, rs) => (itemBlockId, rs.groupBy(_._2).map { case (itemId, s) => (itemId, s.map { case (u, _, r) => (u, r)})})}
          (userBlockId, (userMap, subRatingByUser, subRatingByItem))
        }
      }.partitionBy(unifiedPartitioner)

//    ratings.unpersist()

    var totalTime = 0.0D
    var oldUserRatingsBlock = sc.emptyRDD[(Int, (mutable.Map[Int, Array[Score]], Map[Int, Map[Int, Iterable[(Int, Score)]]], Map[Int, Map[Int, Iterable[(Int, Score)]]]))]
    var oldItemsBlock = sc.emptyRDD[(Int, (Int, mutable.Map[Int, Array[Score]]))]

    var newUserRatingsBlock = userRatingsBlock
    var newItemsBlock = itemsBlock

    var totalCount = 0
    val lossAccum = if (isEvaluate) Some(sc.doubleAccumulator("loss")) else None
    for (iter <- 0.until(numIter)) {
      val start = System.currentTimeMillis() / 1000
      for (_ <- 0.until(numPartition)) {
        val iterResult = newUserRatingsBlock.join(newItemsBlock, newUserRatingsBlock.partitioner.get).mapValues { case ((u, ru, ri), i) =>
          val userMap: mutable.Map[Int, Array[Score]] = u
          val itemBlockId = i._1
          val itemMap = i._2
          val ratingByUser: Map[Int, Iterable[(Int, Score)]] = ru.getOrElse(itemBlockId, Map.empty)
          val ratingByItem: Map[Int, Iterable[(Int, Score)]] = ri.getOrElse(itemBlockId, Map.empty)

          val coff = 1.0F - 2 * lambda_v * alpha

          val newUserMap: mutable.Map[Int, Array[Score]] = userMap.map { case (userId, wi) =>
            val itemsRating: Iterable[(Int, Score)] = ratingByUser.get(userId).map(_.filter { case (id, _) => itemMap.contains(id)}).getOrElse(Iterable.empty)
            if (itemsRating.nonEmpty) {
              val V: Array[Score] = itemsRating.map(_._2).toArray // will change in place
              val H: Array[Score] = itemsRating.flatMap { case (itemId, _) => itemMap(itemId) }.toArray
              val Hx = numFactor
              val Hy = itemsRating.size
              // y <- rating, A <- items, x <- userFactor
              blas.sgemv("T", Hx, Hy, -1.0F, H, Hx, wi, 1, 1.0F, V, 1)
              blas.sgemv("N", Hx, Hy, 2.0F, H, Hx, V, 1, coff, wi, 1)
            }
            (userId, wi)
          }

          val newItemMap = itemMap.map { case (itemId, hj) =>
            val userRating = ratingByItem.getOrElse(itemId, Map.empty)
            if (userRating.nonEmpty) {
              val V = userRating.map(_._2).toArray // will change in place
              val W = userRating.flatMap { case (userId, _) => userMap(userId) }.toArray
              val Wx = numFactor
              val Wy = userRating.size

              blas.sgemv("T", Wx, Wy, -1.0F, W, Wx, hj, 1, 1.0F, V, 1)
              blas.sgemv("N", Wx, Wy, 2.0F, W, Wx, V, 1, coff, hj, 1)

              lossAccum.foreach(accum => {
                val loss = userRating.map(_._2).toArray
                blas.sgemv("T", Wx, Wy, -1.0F, W, Wx, hj, 1, 1.0F, loss, 1)
                val result = blas.snrm2(loss.length, loss, 1)
                accum.add(result.toDouble)
              })
            }
            (itemId, hj)
          }
          ((newUserMap, ru, ri), (unifiedPartitioner.getPartition(itemBlockId + 1), newItemMap)) // rotation
        }

        oldUserRatingsBlock = newUserRatingsBlock
        oldItemsBlock = newItemsBlock

        iterResult.cache()
        newUserRatingsBlock = iterResult.mapValues { case (u, _) => u }
        newItemsBlock = iterResult.mapValues { case (_, i) => i }

        newUserRatingsBlock.cache()
        newItemsBlock.cache()

        totalCount += 1
        if (totalCount % checkPointInterval == 0) {
          newUserRatingsBlock.checkpoint()
          newItemsBlock.checkpoint()
        } else {
          newUserRatingsBlock.count()
          newItemsBlock.count()
        }

        iterResult.unpersist()
        oldUserRatingsBlock.unpersist()
        oldItemsBlock.unpersist()
      }
      val elapsedTime = System.currentTimeMillis() / 1000 - start
      totalTime += elapsedTime
      println(s"loss numFactor $numFactor numPartition $numPartition in iter $iter time $elapsedTime totalTime $totalTime = ${lossAccum.map(a => a.value / numPartition)}")
      lossAccum.foreach(a => a.reset())
    }

    deleteCheckFile(checkFile)
    spark.stop()
    println("Done")
  }
}
