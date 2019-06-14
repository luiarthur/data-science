import scala.util.Random.{setSeed, shuffle}

setSeed(42)

val A = 'A'.toInt
val M = 'M'.toInt
val Z = 'Z'.toInt

object Reflector {
  val state = ((A to M) ++ (A to M)).toVector
}

case class Rotar(var state:Int=A) {
  def toChar = state.toChar
  def advance = {
    val spill = (state == Z)
    if (spill) {
      state = A
    } else {
      state += 1
    }
    spill
  }
}

case class PlugBoard() {
}

case class Enigma(rotars: Vector[Rotar], plugBoard:PlugBoard, seed:Vector[Int]) {

  val numRotars = rotars.size
  require(numRotars > 0)
  require(numRotars == seed.size)
  val idxLoop = numRotars match {
    case 1 => List(0)
    case _ => List.range(0, numRotars) ++ List(0)
  }

  // advance the rotars
  def advance(idx: List[Int]=idxLoop): Unit = {
    if (idx.size > 0) {
      val spill = rotars(idx.head).advance
      if (spill) advance(idx.tail)
    }
  }

  def encode() = ???
}

val e = Enigma(Vector(Rotar(), Rotar(), Rotar()), PlugBoard(), seed=Vector(A, A, A))
for (i <- 1 to 100) {
  e.advance()
  println(e)
}
