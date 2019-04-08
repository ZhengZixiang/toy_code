# BiLSTM-CRF for Sequence Labelling

基于PyTorch实现的用于序列标注任务的BiLSTM-CRF模型，代码参考自[threelittlemonkeys/lstm-crf-pytorch](https://github.com/threelittlemonkeys/lstm-crf-pytorch)。原始代码有点小问题，这里补全了中文分词的word2id。

## 运行代码
数据存放于data目录下，数据格式如下：
```
今天 天气 不错 ...
``` 

准备数据：
```commandline
python prepare.py ./data/training_data
```

训练：
```commandline
python train.py ./model/model_name ./data/char2id ./data/word2id ./data/tag2id ./data/training_data.csv num_epoch
```

预测：
```commandline
python predict.py ./model/model_name.epochN ./data/word2id ./data/tag2id ./data/test_data
```
