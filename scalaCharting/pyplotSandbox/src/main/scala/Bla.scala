import com.github.sh0nk.matplotlib4j
import scala.collection.JavaConverters._ // .asJava

object Bla {
  println("Here")
  val rng = new scala.util.Random(0)

  // See example here: https://github.com/sh0nk/matplotlib4j

  // Scatter plot
  val plt = matplotlib4j.Plot.create()
  plt.title("Bob")
  val n = 1000
  val x = List.tabulate(n)( _ => rng.nextGaussian ).map{ Double.box }.asJava
  val y = List.tabulate(n)( _ => rng.nextGaussian ).map{ Double.box }.asJava
  plt.plot.add(x, y, "o")
  // plt.show() // shows plot immediately
  plt.savefig("here.pdf") // plots to project top dir
  plt.executeSilently() // necessary to make plot

  // Heatmap?
  //val plt2 = matplotlib4j.Plot.create()
  //val xx = List.range(0, 5)
  //val yy = List.range(0, 5)
  //val z = for (x <- xx; y <- yy) yield x + y + 1.0
  //val xyT = xy.transpose.map{ _.map{zz => Double.box(zz)}.toList.asJava }
  //plt2.pcolor.add(xx, yy, z).cmap("plt.cm.Blues")
}
