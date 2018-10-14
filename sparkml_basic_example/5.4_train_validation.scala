import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)

//训练验证分离实例
//准备训练和测试数据
//terminal: sed -i 's/+//g' sample_linear_regression_data.txt
val data = sqlContext.read.format("libsvm").load("/home/zhengzx/Desktop/sample_linear_regression_data.txt")

val Array(training, testing) = data.randomSplit(Array(0.9, 0.1), seed=12345)

//使用ParamGridBuilder构造一个参数网络
val lr = new LinearRegression()
val paramGrid = new ParamGridBuilder()
.addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
.addGrid(lr.fitIntercept)
.addGrid(lr.regParam, Array(0.1, 0.01))
.build()

//使用TrainValidationSplit来选择模型和参数
val tv = new TrainValidationSplit().
setEstimator(lr).
setEstimatorParamMaps(paramGrid).
setEvaluator(new RegressionEvaluator()).
setTrainRatio(0.8)

//运行训练验证分离，选择最好的参数
val model = tv.fit(training)

//在测试数据上做预测，模型是参数组合中执行最好的一个
model.transform(testing).select("features", "label", "prediction").show()
