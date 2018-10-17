package me.zhengzx.rec

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

object MovieLensALS {

  //1. Define a rating elicitation function
  def elicitRating(movies: Seq[(Int, String)]) = {
    val prompt = "Please rate the following movie (1-5(best) or 0 if not seen):"
    println(prompt)
    val ratings = movies.flatMap { x =>
      var rating: Option[Rating] = None
      var valid = false
      while (!valid) {
        println(x._2 + " :")
        try {
          var r = Console.readInt()
          if (r > 5 || r < 0) {
            println(prompt)
          } else {
            valid = true
            if (r > 0) {
              rating = Some(Rating(0, x._1, r))
            }
          }
        } catch {
          case e: Exception => println(prompt)
        }
      }
      rating match {
        case Some(r) => Iterator(r)
        case None => Iterator.empty
      }
    }
    if (ratings.isEmpty) {
      sys.error("No ratings provided!")
    } else {
      ratings
    }
  }

  //2. Define a RMSE computation function
  def computeRMSE(model: MatrixFactorizationModel, data: RDD[Rating]) = {
    val prediction = model.predict(data.map(x => (x.user, x.product)))
    val predDataJoined = prediction.map(x => ((x.user, x.product), x.rating)).join(data.map(x => ((x.user, x.product), x.rating))).values
    new RegressionMetrics(predDataJoined).rootMeanSquaredError
  }

  //3. main
  def main(args: Array[String]) {
    //3.1 Setup env
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    if(args.length != 1) {
      print("Usage: MovieLensHomeDir")
      sys.exit(1)
    }

    val conf = new SparkConf().setAppName("MovieLensALS").set("spark.executor.memory", "1024m")
    val sc = new SparkContext(conf)

    //3.2 Load ratings data and know your data
    val movieLensHomeDir = args(0)
    val ratings = sc.textFile(new File(movieLensHomeDir, "ratings.dat").toString).map{line =>
      val fields = line.split("::")
      //timestamp, user, product, rating
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }
    val movies = sc.textFile(new File(movieLensHomeDir, "movies.dat").toString).map{line =>
      val fields = line.split("::")
      //movieId, movieName
      (fields(0).toInt, fields(1))
    }

    val numRatings = ratings.count()
    val numUsers = ratings.map(x => x._2.user).distinct().count()
    val numMovies = ratings.map(x => x._2.product).distinct().count()
    println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.")

    //3.3 Elicitate personal rating
    val topMovies = ratings.map(_._2.product).countByValue().toSeq.sortBy(-_._2).take(50).map(_._1)
    val random = new Random(0)
    val selectMovies = topMovies.filter(x => random.nextDouble() < 0.2).map(x => (x, movies.lookup(x)(0)))
    val myRatings = elicitRating(selectMovies)
    val myRatingsRDD = sc.parallelize(myRatings, 1)

    //3.4 Split data into train(60%), validation(20%) and test(20%)
    val numPartitions = 10
    val trainSet = ratings.filter(x => x._1 < 6).map(_._2).union(myRatingsRDD).repartition(numPartitions).persist()
    val validationSet = ratings.filter(x => x._1 >= 6 && x._1 < 8).map(_._2).persist()
    val testSet = ratings.filter(x => x._1 >= 8).map(_._2).persist()

    val numTrain = trainSet.count()
    val numValidation = validationSet.count()
    val numTest = testSet.count()
    println("Training data: " + numTrain + " Validation data: " + numValidation + " Test data: " + numTest)

    //3.5 Train model and optimize model with validation set
    val numRanks = List(8, 12)
    val numIters = List(10, 20)
    val numLambdas = List(0.1, 10.0)
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestRank = -1
    var bestIter = -1
    var bestLambda = -1.0
    var bestRMSE = Double.MaxValue
    for(rank <- numRanks; iter <- numIters; lambda <- numLambdas) {
      val model = ALS.train(trainSet, rank, iter, lambda)
      val validationRMSE = computeRMSE(model, validationSet)
      println("RMSE(validation)=" + validationRMSE + " with rank=" + rank + ", iter=" + iter + ", lambda=" + lambda)
      if(validationRMSE < bestRMSE) {
        bestModel = Some(model)
        bestRank = rank
        bestIter = iter
        bestLambda = lambda
        bestRMSE = validationRMSE
      }
    }

    //3.6 Evaluate model on test set
    val testRMSE = computeRMSE(bestModel.get, testSet)
    println("The best model was trained with rank=" + bestRank + ", iter=" + bestIter + ", lambda=" + bestLambda +
      "and compute RMSE on test set is " + testRMSE)


    //3.7 Create a baseline and evaluate model with test set
    val meanRating = trainSet.union(validationSet).map(_.rating).mean()
    val baselineRMSE = new RegressionMetrics(testSet.map(x => (x.rating, meanRating))).rootMeanSquaredError
    val improvement = (baselineRMSE - testRMSE) / baselineRMSE * 100
    println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    //3.8 Make a personal recommendation
    val candidates = movies.keys.filter(!myRatings.contains(_))
    val recommendations = bestModel.get
      .predict(candidates.map(x => (0, x)))
      .sortBy(-_.rating)
      .take(50)

    var i = 0
    println("Movies recommended for you")
    recommendations.foreach {line =>
      println("%2d".format(i) + ": " + movies.lookup(line.product)(0))
      i += 1
    }


    sc.stop()
  }

}
