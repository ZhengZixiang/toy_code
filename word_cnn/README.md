代码使用TensorFlow高阶API创建一个基于Estimator接口的TensorFlow模型，模型

## 运行代码
下载数据后先建立语料字典
```python
python build_vocab.py
```
训练，可参考代码中flags来指定各种超参数
```python
python model.py
```

## 数据集
所使用的数据集为[DBPedia Ontology Classification Dataset](https://raw.githubusercontent.com/srhrshr/torchDatasets/master/dbpedia_csv.tar.gz)，目标是从DBPedia 2014数据集中对文本进行分类，共有14个互不重叠类别。每个类别随机选择了40,000个训练样本和5,000个测试样本。因此，总共有560,000个训练样本和70,000个测试样本。因此，总共有560

## 模型结构
本代码参考经典的CNN文本分类模型实现，如下图所示。
![](https://pic1.zhimg.com/80/v2-49cd389458cd6a8a0dc4babc06e712dc_hd.jpg)

将文本抽取出高频字典后，将所有文本截断或补长为100个单词，再喂给模型。模型流程如下：
1. 将训练数据经过embedding_lookup转化成ids序列。
2. 经过三种大小(3,100)，(4,100), (5,100)的卷积核进行conv2d并池化。
3. flatten成dense layer。
4. softmax输出分类。
![](https://pic4.zhimg.com/80/v2-66ed2ca5cec5ee35c4ea1e8cdb467453_hd.jpg)

## 参考文献
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)