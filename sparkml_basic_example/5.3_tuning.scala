import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{Tokenizer, HashingTF}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)

//模型调优实例
//准备训练数据
val training = sqlContext.createDataFrame(Seq(
 (0L, "a b c d e spark", 1.0),
 (1L, "b d", 0.0),
 (2L, "spark f g h", 1.0),
 (3L, "hadoop mapreduce", 0.0),
 (4L, "b spark who", 1.0),
 (5L, "g d a y", 0.0),
 (6L, "spark fly", 1.0),
 (7L, "was mapreduce", 0.0),
 (8L, "e spark program", 1.0),
 (9L, "a e c l", 0.0),
 (10L, "spark compile", 1.0),
 (11L, "hadoop software", 0.0)
)).toDF("id", "text", "label")

//配置ML管道，包含三个stage: Tokenizer, HashingTF, lr
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("features")
val lr = new LogisticRegression().setMaxIter(10)
val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

//使用ParamGridBuilder构造一个参数网格
val paramGrid = new ParamGridBuilder().
addGrid(hashingTF.numFeatures, Array(10, 100, 1000)).
addGrid(lr.regParam, Array(0.1, 0.01)).
build()

//使用CrossValidator来选择模型和参数
//CrossValidator需要一个Estimator，一个评估器参数集合，一个Evaluator
val cv = new CrossValidator()
.setEstimator(pipeline)
.setEstimatorParamMaps(paramGrid)
.setEvaluator(new BinaryClassificationEvaluator())
.setNumFolds(2)

//运行交叉验证，选择最好的参数集
val cvModel = cv.fit(training)

//准备测试数据
val testing = sqlContext.createDataFrame(Seq(
 (12L, "spark h d e"),
 (13L, "a f c"),
 (14L, "mapreduce spark"),
 (15L, "apache hadoop")
)).toDF("id", "text")

//预测结果
cvModel.transform(testing).select("id", "text", "probability", "prediction").collect().
foreach{case Row(id: Long, text: String, probability: Vector, prediction: Double) => println(s"($id, $text) -> probability=$probability, prediction=$prediction")}
