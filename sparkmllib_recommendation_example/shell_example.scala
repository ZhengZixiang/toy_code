import org.apache.spark.sql.types._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)

import sqlContext.implicits._

case class Movie(movieId: Int, title: String, genres: Seq[String])
case class User(userId: Int, gender: String, Age: Int, occupation: Int, zip: String)

//Define parse function
def parseMovie(str: String): Movie = {
 val fields = str.split("::")
 assert(fields.size == 3)
 Movie(fields(0).toInt, fields(1).toString, Seq(fields(2)))
}
def parseUser(str: String): User = {
 val fields = str.split("::")
 assert(fields.size == 5)
 User(fields(0).toInt, fields(1).toString, fields(2).toInt, fields(3).toInt, fields(4).toString)
}
def parseRating(str: String): Rating = {
 val fields = str.split("::")
 assert(fields.size == 4)
 Rating(fields(0).toInt, fields(1).toInt, fields(2).toInt)
}

//Ratings analyst
val ratingText = sc.textFile("file:/data/ml-1m/ratings.dat")
ratingText.first()
//cache持久化到内存中
val ratingRDD = ratingText.map(parseRating).cache()
//输出评分记录数
println("Total number of ratings: " + ratingRDD.count())
//输出被评分电影数
println("Total number of movies rated: " + ratingRDD.map(_.product).distinct().count())
//输出已评分用户数
println("Total number of users who rated movies: " + ratingRDD.map(_.user).distinct().count())

//Create DataFrame
val ratingDF = ratingRDD.toDF();
//查看DF是否创建成功
ratingDF.printSchema()
//同理创建movieDF、userDF
val movieDF = sc.textFile("file:/data/ml-1m/movies.dat").map(parseMovie).toDF()
movieDF.printSchema()
val userDF = sc.textFile("file:/data/ml-1m/users.dat").map(parseUser).toDF()
userDF.printSchema()
//转化成SQL表
ratingDF.registerTempTable("ratings")
movieDF.registerTempTable("movies")
userDF.registerTempTable("users")

//查询每部电影的评分情况
val result = sqlContext.sql("""select title, rmax, rmin, ucnt from
(select product, max(rating) as rmax, min(rating) as rmin, count(distinct user) as ucnt
from ratings group by product) ratingsCNT
join movies on product=movieId
order by ucnt desc""")
//展示前20行数据
result.show()

//查询最活跃的用户（即评分过的电影最多的用户） count(user) count(*)均可
val mostActiveUser = sqlContext.sql("""select user, count(user) as cnt
from ratings group by user order by cnt desc limit 10""")
mostActiveUser.show()

//查询最活跃用户的评分大于4的电影名称与数量
val result = sqlContext.sql("""select title, rating
from ratings join movies on product=movieId
where user=4169 and rating > 4""")
result.show()
result.count()

//ALS
val splits = ratingRDD.randomSplit(Array(0.8, 0.2), 0L)
val trainSet = splits(0).cache()
val testSet = splits(1).cache()
trainSet.count()
testSet.count()
//rank->隐因子
val model = (new ALS().setRank(20).setIterations(10).run(trainSet))

//为最活跃用户推荐电影
val recomForTopUser = model.recommendProducts(4169, 5)
//取出电影ID到title的键值Map，由于Spark API变化，下面这句无法实现，必须经过一些变换
//val movieTitle = movieDF.map(array=>(array(0),array(1))).collectAsMap()
val collect = movieDF.select("movieId", "title").collect
import scala.collection.mutable.Map 
val movieTitle:Map[Int, String] = Map()
x = 0
for(x <- collect) {
 val movieId = x(0).asInstanceOf[Int]
 val title = x(1).asInstanceOf[String]
 movieTitle += (movieId -> title)
}

//根据Map的ID映射出所推荐的电影
val recomResult = recomForTopUser.map(rating => (movieTitle(rating.product), rating.rating)).foreach(println)

val testUserProduct = testSet.map{
 case Rating(user, product, rating) => (user, product)
}
//预测
val testUserProductPredict = model.predict(testUserProduct)
testUserProductPredict.take(10).mkString("\n")

//模型评估
//先尝试自己撸一个MAE
//将test的(user, product)作为key来做预测值和实际值的比较
val testSetPair = testSet.map{
 case Rating(user, product, rating) =>
 ((user, product), rating)
}
val predictionPair = testUserProductPredict.map{
 case Rating(user, product, rating) =>
 ((user, product), rating)
}
//连接两个Map
val joinTestPredict = testSetPair.join(predictionPair)
joinTestPredict.take(10).foreach(println)
//求平均误差
val mae = joinTestPredict.map{
 case ((user, product), (ratingT, ratingP)) =>
 val err = ratingT - ratingP
 Math.abs(err)
}.mean()

//对于0 1型的混淆矩阵，易于处理
//对于数值型的混淆矩阵，FP我们一般采用：ratingT<=1, ratingP>=4
val fp = joinTestPredict.filter{
 case ((uesr, product), (ratingT, ratingP)) =>
 (ratingT <= 1 & ratingP >= 4)
}

//再使用API提供的评估指标
import org.apache.spark.mllib.evaluation._
val ratingTP = joinTestPredict.map {
 case ((user, product), (ratingT, ratingP)) =>
 (ratingP, ratingT)
}
val evaluator = new RegressionMetrics(ratingTP)
evaluator.meanAbsoluteError
evaluator.rootMeanSquaredError

