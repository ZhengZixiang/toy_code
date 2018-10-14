import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{Tokenizer, HashingTF}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)

//准备训练文档 seq: id, text ...
val training = sqlContext.createDataFrame(Seq(
 (0L, "a b c d e spark", 1.0),
 (1L, "b d", 0.0),
 (2L, "spark f g h", 1.0),
 (3L, "hadoop mapreduce", 0.0)
)).toDF("id", "text", "label")

//配置ML管道，包含三个stage: 分词Tokenizer, 词频Hashing TF, 模型lr: Transformer->Estimator
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).
setOutputCol("features").setNumFeatures(1000)

val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01)

val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

//安装管道到数据上
val model = pipeline.fit(training)

//保存管道到磁盘，包括安装好的和未安装好的
pipeline.save("/data/sparkml_lrpipeline")
model.save("/data/sparkml_lrmodel")

//加载管道
val model2 = PipelineModel.load("/data/sparkml_lrmodel")

//准备测试文档
val testing = sqlContext.createDataFrame(Seq(
 (4L, "spark h d e"),
 (5L, "a f c"),
 (6L, "mapreduce spark"),
 (7L, "apache hadoop")
)).toDF("id", "text")

//预测结果
model.transform(testing).select("id", "text", "probability", "prediction").collect().
foreach{case Row(id: Long, text: String, probability: Vector, prediction: Double) => println(s"($id, $text) -> probability=$probability, prediction=$prediction")}
