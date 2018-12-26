使用TensorFlow高阶API创建基于Estimator、Dataset等接口的Wide & Deep Learning模型，实现收入情况二分类预估。

## 运行代码
数据文件存放在`wdl_data`，数据集关于[收入情况分析](http://mlr.cs.umass.edu/ml/machine-learning-databases/adult)，有30000多个训练样本。

第一次运行，需要使用`maybe_download`函数可以下载数据，后参考`__main__`的参数指定数据文件。

这里提供的原始数据有一点小问题，第一次训练模型务必在`main`函数中调用`clean`函数，清洗完数据后，以后训练模型注释掉即可。
```python
python model.py
```
参数指定参考代码`FLAGS`。模型架构大致如下：
![](https://1.bp.blogspot.com/-Dw1mB9am1l8/V3MgtOzp3uI/AAAAAAAABGs/mP-3nZQCjWwdk6qCa5WraSpK8A7rSPj3ACLcB/s1600/image04.png)
from Google AI Blog

## 参考文献
[Google.Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)