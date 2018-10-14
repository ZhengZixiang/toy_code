import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)	

//准备带标签和特征的数据 DataFrame
val training = sqlContext.createDataFrame(Seq(
 (1.0, Vectors.dense(0.0, 1.1, 0.1)),
 (0.0, Vectors.dense(2.0, 1.0, -1.0)),
 (0.0, Vectors.dense(2.0, 1.3, 1.0)),
 (1.0, Vectors.dense(0.0, 1.2, -0.5))
)).toDF("label", "features")

//观察是否创建DF成功
training.take(2)

//创建逻辑回归分类器
val lr = new LogisticRegression()

//解释参数
println(lr.explainParams())
//使用setter方法设置参数：迭代次数和正则化
lr.setMaxIter(10).setRegParam(0.01)
//再输出一次参数，观察是否修改成功
println(lr.explainParams())

//使用存储在lr中的参数来训练一个模型
val model1 = lr.fit(training)
//查看模型参数
model1.parent.extractParamMap

//使用ParamMap设置参数
val paramMap = ParamMap(lr.maxIter -> 20).
put(lr.maxIter, 30).
put(lr.regParam -> 0.1, lr.threshold -> 0.55)
//ParamMap组合
val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")
val paramMapCombined = paramMap ++ paramMap2

//通过ParamMap指定的参数来训练一个模型
val model2 = lr.fit(training, paramMapCombined)
//查看模型参数
model2.parent.extractParamMap

//准备测试数据
val testing = sqlContext.createDataFrame(Seq(
 (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
 (0.0, Vectors.dense(3.0, 2.0, -0.1)),
 (1.0, Vectors.dense(0.0, 2.2, -1.5))
)).toDF("label", "features")

//预测
model1.transform(testing).collect  // take
//格式化输出
model1.transform(testing).select("label", "features", "probability", "prediction").collect.
foreach{case Row(label: Double, features: Vector, probability: Vector, prediction: Double) => println(s"($features, $label) -> probability=$probability, prediction = $prediction")}
