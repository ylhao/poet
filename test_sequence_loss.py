# coding: utf-8


import numpy as np
import tensorflow as tf


# 2 * 3 * 4
# 可以看成 batch_size = 2, max_time = 3, len(char2id) + 1 = 4
logits_np = np.array([
    [[0.6, 0.5, 0.3, 0.2], [0.9, 0.5, 0.3, 0.2], [1.0, 0.5, 0.3, 0.2]],
    [[0.2, 0.5, 0.3, 0.2], [0.3, 0.5, 0.3, 0.2], [0.4, 0.5, 0.3, 0.2]]
])

targets_np = np.array([
    [0, 1, 2],
    [3, 0, 1]
], dtype=np.int32)

logits = tf.convert_to_tensor(logits_np)
targets = tf.convert_to_tensor(targets_np)

cost = tf.contrib.seq2seq.sequence_loss(logits=logits,
                     targets=targets,
                     weights=tf.ones_like(targets, dtype=tf.float64))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    r = sess.run(cost)
    print(r)

print((np.log(np.exp(0.6) / (np.exp(0.6) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2))) + 
       np.log(np.exp(0.5) / (np.exp(0.9) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2))) + 
       np.log(np.exp(0.3) / (np.exp(1.0) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2))) + 
       np.log(np.exp(0.2) / (np.exp(0.2) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2))) + 
       np.log(np.exp(0.3) / (np.exp(0.3) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2))) + 
       np.log(np.exp(0.5) / (np.exp(0.4) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2)))) / 6)

"""
tf.contrib.seq2seq.sequence_loss可以直接计算序列的损失函数，重要参数：
    logits：尺寸[batch_size, sequence_length, num_decoder_symbols]
    targets：尺寸[batch_size, sequence_length]，不用做one_hot。
    weights：[batch_size, sequence_length]，即mask，滤去padding的loss计算，使loss计算更准确。

通过上例可以看出：
tf.contrib.seq2seq.sequence_loss的计算过程大致如下：
先算 softmax
然后计算所有对应位置的 softmax 值得和
最后除上 (batch_size * sequence_length)
取负
"""
