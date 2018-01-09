import scala.io.Source

object Recommender {
  case class Movie(id:Int, title:String, genre:Set[String])

  case class Review(userId:Int, movieId:Int, score:Double) {
    def classify() = {
      require(score >= 0 && score <=5)
      score match {
        case x if x <= 2.5 => -1
        case x if x >= 3.5 => 1
        case _ => 0
      }
    }

    def similar(that:Review):Boolean = {
      this.movieId == that.movieId && this.classify == that.classify
    }
  }

  val fileDir = "../data/ml-latest-small/"

  def readFile(file:String) = Source.fromFile(fileDir + file).getLines

  val movies = readFile("movies.csv").toList.tail.map{ line =>
    val s = line.split(',')
    Movie(s(0).toInt, s(1), s(2).split('|').toSet)
  }.toSet

  val reviews = readFile("ratings.csv").toList.tail.map{ line =>
    val s = line.split(',')
    Review(s(0).toInt, s(1).toInt, s(2).toDouble)
  }

  /* Movie IDs:
   * - Toy Story: 2253
   * - Batman Begins: 33794
   */

  /*
  val review = Review(0, 33794, 4.5)
  */

  def recommend(review:Review, numMovies:Int=3, atLeast:Double=4) = {
    val similarReviews = reviews.filter(_.similar(review))
    val similarUsers = relevantReviews.map(_.userId).distinct
    //val relevantReviews = reviews.filter()
    ???
  }
}
