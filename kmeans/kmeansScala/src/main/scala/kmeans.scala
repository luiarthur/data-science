package kmeansScala
import scala.util.Random
import math.{sqrt, pow}

object Kmeans {
  def timer[R](block: => R) = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1E9 + "s")
    result
  }

  def addVec(x:Array[Double], y:Array[Double]):Array[Double] = {
    // TODO: Make more efficient
    x.zip(y).map(xy => xy._1 + xy._2)
  }

  def colMeans(X:Array[Array[Double]]):Array[Double] = {
    // TODO: Make more efficient
    val n = X.size
    X.reduceLeft((l,r) => addVec(l,r)).map( _ / n )
  }

  def minind(x:Array[Double]):Int = {
    // TODO: Make more efficient
    x.zipWithIndex.minBy(_._1)._2
  }

  def dist(x:Array[Double], y:Array[Double]) = {
    var i = 0
    val n = x.size
    var d = 0.0

    while (i < n) {
      d += pow(x(i) - y(i), 2)
      i += 1
    }

    sqrt(d)
  }

  def distMatrix(X:Array[Array[Double]], Y:Array[Array[Double]]):Double = {
    sqrt(X.zip(Y).map{ case(x,y) => pow(dist(x,y),2) }.sum)
  }

  def kmeans(X: Array[Array[Double]], K:Int, iterMax:Int=100, eps:Double=1E-5) = {
    val n = X.size
    val p = X.head.size
    val indices = Array.range(0,n)


    def updatedLabels(centers: Array[Array[Double]]):Array[Int] = {
      // TODO: parallelize?
      indices.par.map{ i => 
        minind(centers.map{ c => dist(X(i), c) }) 
      }.toArray
      
      //indices.foreach{ i => 
      //  labels(i) = minind(centers.map{ c => dist(X(i), c) }) 
      //}
    }


    def updatedCenters(labels: Array[Int]):Array[Array[Double]] = {
      // TODO: parallelize?
      Array.range(0,K).par.map{ k =>
        colMeans(labels.zipWithIndex.filter(_._1 == k).map{ case(_,idx) => X(idx) })
      }.toArray
    }

    val init:Array[Array[Double]] = Random.shuffle(X.toList).take(K).toArray
    var iter = 0 

    def optim(centers:Array[Array[Double]], labels:Array[Int]=Array.fill(0)(n)):Array[Int] = {
      iter += 1
      val newLabels = updatedLabels(centers)
      val newCenters = updatedCenters(newLabels)
      if (distMatrix(newCenters, centers) < eps || iter > iterMax) {
        labels
      } else optim(newCenters, newLabels)
    }

    val labels = optim(init)
    val centers = updatedCenters(labels)

    (centers, labels, iter)
  }
}
