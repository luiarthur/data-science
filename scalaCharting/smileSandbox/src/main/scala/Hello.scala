import smile.plot._

// https://haifengl.github.io/smile/visualization.html#scatter
object Hello {
  // the matrix to display
  val z = Array(
    Array(1.0, 2.0, 4.0, 1.0),
    Array(6.0, 3.0, 5.0, 2.0),
    Array(4.0, 2.0, 1.0, 5.0),
    Array(5.0, 4.0, 2.0, 3.0)
  )

  // make the matrix larger with bicubic interpolation
  val x = Array(0.0, 1.0, 2.0, 3.0)
  val y = Array(0.0, 1.0, 2.0, 3.0)
  //val bicubic = new BicubicInterpolation(x, y, z)
  val Z = Array.ofDim[Double](101, 101)
  for (i <- 0 to 100) {
    for (j <- 0 to 100)
      Z(i)(j) = scala.util.Random.nextGaussian //bicubic.interpolate(i * 0.03, j * 0.03)
  }

  heatmap(Z, Palette.jet(256))

  //val canvas = Heatmap.plot(Z, Palette.jet(256))
  //val headless = new Headless(canvas)
  //headless.pack
  //headless.setVisible(true)

  //canvas.save(new java.io.File("headless.png"))
}
