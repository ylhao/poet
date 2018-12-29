# POET

这是一个基于深度学习自动生成古诗词的项目。参考《深度有趣：歌词古诗自动生成》，使用的数据集来自于[中华古诗词数据库](https://github.com/chinese-poetry/chinese-poetry)，数据集提供[下载](https://pan.baidu.com/s/1GC1ktbP1yo-oERvFR_CUYw)。

## 重要知识点

- TensorFlow
- SnowNLP
- LSTM
- 静态 RNN 和动态 RNN

## 重要 API

- tf.nn.rnn_cell.BasicLSTMCell
- tf.nn.rnn_cell.MultiRNNCell
- tf.nn.dynamic_rnn
- tf.contrib.seq2seq.sequence_loss

## 脚本

- train.py: 训练模型、存储模型
- gen.py: 生成古诗词
- test_sequence_loss.py: 了解 tf.contrib.seq2seq.sequence_loss 的计算方式

## 结果

```
同谈南北辅潮来，苏孟宗儒拔不传。柱下抚陈形相显，严禋端可慑兰房。
书生绝咽相驱路，生死投閒定颇林。意淡自难求我好，气清如此拜南屏。[藏头诗：书生意气]
```
