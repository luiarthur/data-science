import scala.io.Source

object Recommender {
  case class Movie(id:Int, title:String, genre:Set[String])

  case class Review(userId:Int, movieId:Int, score:Double) {
    def classify() = {
      require(score >= 0 && score <=5)
      score match {
        case x if x <= 2 => -1
        case x if x >= 4 => 1
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

  val movieMap = movies.map(m => (m.id, m.title)).toMap

  val reviews = readFile("ratings.csv").toList.tail.map{ line =>
    val s = line.split(',')
    Review(s(0).toInt, s(1).toInt, s(2).toDouble)
  }


  def recommend(review:Review, numMovies:Int=3) = {
    val similarReviews = reviews.filter(_.similar(review))
    val similarUsers = similarReviews.map(_.userId).distinct
    val relevantReviews = reviews.filter(r => similarUsers.contains(r.userId))

    val counts = collection.mutable.Map[Int,Int]().withDefaultValue(0)
    val scores = collection.mutable.Map[Int,Double]().withDefaultValue(0)

    relevantReviews.foreach{ r => 
      counts(r.movieId) += 1 
      scores(r.movieId) += r.score
    }
    scores.keys.foreach{ k => scores(k) /= counts(k) }

    //scores.foreach(println)
    scores.toList.sortBy(-_._2).take(numMovies).map(m => (movieMap(m._1),m._2))
  }
}

/* Movie IDs:
 * - Toy Story: 2253
 * - Batman Begins: 33794
 */

/* Test
import Recommender._
recommend(Review(0, 33794, 0), 10).foreach{println}
recommend(Review(0, 33794, 1), 10).foreach{println}
recommend(Review(0, 33794, 5), 10).foreach{println}
val x = reviews.filter(_.movieId == 33794).map(_.classify)
*/
