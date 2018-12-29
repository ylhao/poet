# encoding: utf-8


import tensorflow as tf
import numpy as np
import glob
import json
from collections import Counter
from tqdm import tqdm
from snownlp import SnowNLP
import codecs
import pickle


BATCH_SIZE = 64
HIDDEN_SIZE = 256  # 隐层单元数
NUM_LAYER = 2  # 层数
EMBEDDING_SIZE = 256  # 嵌入层单元数


def load_data():
    poets = []
    """
    1. glob.glob函数的参数是字符串。这个字符串的书写和我们使用linux的shell命令相似，或者说基本一样。
    也就是说，只要我们按照平常使用cd命令时的参数就能够找到我们所需要的文件的路径。
    字符串中可以包括“*”、“?”和"["、"]"，其中“*”表示匹配任意字符串，“?”匹配任意单个字符，[0-9]与[a-z]表示匹配0-9的单个数字与a-z的单个字符。
    2.glob.glob不支持“~”波浪符号，这个符号在linux代表当前用户的home目录。
    """
    paths = glob.glob('chinese-poetry/json/poet.*.json')
    for path in paths:
        data = codecs.open(path, 'r', encoding='utf-8').read()
        data = json.loads(data)  # 解析 json 数据
        for item in data:
            content = ''.join(item['paragraphs'])
            if len(content) >= 24 and len(content) <= 32:
                content = SnowNLP(content)
                poets.append('[' + content.han + ']')

    poets.sort(key=lambda x: len(x))
    print('共 {} 首诗'.format(len(poets)))
    print('poets[0]:', poets[0]) 
    print('poets[-1]:', poets[-1])
    return poets


def make_dict():
    """
    构建字典
    """
    chars = []
    for item in poets:
        chars += [c for c in item]
    print('共 {} 个字'.format(len(chars)))
    chars = sorted(Counter(chars).items(), key=lambda x:x[1], reverse=True)  # 返回一个列表
    print('共%d个不同的字' % len(chars))
    print(chars[:10])
    chars = [c[0] for c in chars]
    char2id = {c: i + 1 for i, c in enumerate(chars)}
    id2char = {i + 1: c for i, c in enumerate(chars)}
    return char2id, id2char


def make_batches():
    X_data = []
    Y_data = []
    for b in range(len(poets) // BATCH_SIZE):  # 共有 len(poets) // BATCH_SIZE 个 batch
        start = b * BATCH_SIZE
        end = b * BATCH_SIZE + BATCH_SIZE
        batch = [[char2id[c] for c in poets[i]] for i in range(start, end)]
        maxlen = max(map(len, batch))  # 求一个 batch 中的最大长度
        X_batch = np.full((BATCH_SIZE, maxlen - 1), 0, np.int32)
        Y_batch = np.full((BATCH_SIZE, maxlen - 1), 0, np.int32)
        for i in range(BATCH_SIZE):
            """
            例子：
            x: 来年二月二，与汝暂相弃。烧灰散长江，勿占檀那地
            y: 年二月二，与汝暂相弃。烧灰散长江，勿占檀那地。
            """
            X_batch[i, :len(batch[i]) - 1] = batch[i][:-1]
            Y_batch[i, :len(batch[i]) - 1] = batch[i][1:]
        X_data.append(X_batch)
        Y_data.append(Y_batch)
    print('len(X_data):', len(X_data))
    print('len(Y_data):', len(Y_data))
    return X_data, Y_data


poets = load_data()
char2id, id2char = make_dict()
X_data, Y_data = make_batches()


"""
定义网络结构
"""
X = tf.placeholder(tf.int32, [BATCH_SIZE, None])
Y = tf.placeholder(tf.int32, [BATCH_SIZE, None])
learning_rate = tf.Variable(0.0, trainable=False)
# 定义一个 2 层的 LSTM 结构
cell = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True) for i in range(NUM_LAYER)], state_is_tuple=True)
# 定义初始状态，这里传入的是一个整数，所以返回的是一个 [BATCH_SIZE, state_size] 的 tensor
initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
embeddings = tf.Variable(tf.random_uniform([len(char2id) + 1, EMBEDDING_SIZE], -1.0, 1.0))  # len(char2id) + 1，+1 是因为编号从 1 开始
embedded = tf.nn.embedding_lookup(embeddings, X)
# 使用动态 RNN，outputs: BATCH_SIZE, max_time, HIDDEN_SIZE
outputs, _ = tf.nn.dynamic_rnn(cell, embedded, initial_state=initial_state)
outputs = tf.reshape(outputs, [-1, HIDDEN_SIZE])  # BATCH_SIZE * max_time, HIDDEN_SIZE
logits = tf.layers.dense(outputs, units=len(char2id) + 1)  # BATCH_SIZE * max_time, len(char2id) + 1
logits = tf.reshape(logits, [BATCH_SIZE, -1, len(char2id) + 1])  # BATCH_SIZE, max_time, len(char2id) + 1
loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits, Y, tf.ones_like(Y, dtype=tf.float32)))
params = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5)
optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, params))


# 训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(50):
    sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
    # 打乱数据
    data_index = np.arange(len(X_data))
    np.random.shuffle(data_index)
    X_data = [X_data[i] for i in data_index]
    Y_data = [Y_data[i] for i in data_index]
    losses = []
    for i in tqdm(range(len(X_data))):
        ls_,  _ = sess.run([loss, optimizer], feed_dict={X: X_data[i], Y: Y_data[i]})
        losses.append(ls_)
    print('Epoch %d Loss %.5f' % (epoch, np.mean(losses)))


saver = tf.train.Saver()
saver.save(sess, './poet_generation_tensorflow')
with open('dictionary.pkl', 'wb') as fw:
    pickle.dump([char2id, id2char], fw)
    