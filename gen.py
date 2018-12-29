# coding: utf-8


import tensorflow as tf
import numpy as np
import pickle


# 加载词典
with open('dictionary.pkl', 'rb') as fr:
    [char2id, id2char] = pickle.load(fr)


# 定义参数
BATCH_SIZE = 1
HIDDEN_SIZE = 256
NUM_LAYER = 2
EMBEDDING_SIZE = 256


X = tf.placeholder(tf.int32, [BATCH_SIZE, None])
Y = tf.placeholder(tf.int32, [BATCH_SIZE, None])
learning_rate = tf.Variable(0.0, trainable=False)
cell = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True) for i in range(NUM_LAYER)], 
    state_is_tuple=True)
initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
embeddings = tf.Variable(tf.random_uniform([len(char2id) + 1, EMBEDDING_SIZE], -1.0, 1.0))
embedded = tf.nn.embedding_lookup(embeddings, X)
outputs, last_states = tf.nn.dynamic_rnn(cell, embedded, initial_state=initial_state)
outputs = tf.reshape(outputs, [-1, HIDDEN_SIZE])
logits = tf.layers.dense(outputs, units=len(char2id) + 1)
probs = tf.nn.softmax(logits)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./'))


def generate():
    """
    随机生成一首古诗
    """
    states_ = sess.run(initial_state)
    gen = ''
    c = '['
    while c != ']':
        gen += c
        x = np.zeros((BATCH_SIZE, 1))
        x[:, 0] = char2id[c]
        probs_, states_ = sess.run([probs, last_states], feed_dict={X: x, initial_state: states_})
        probs_ = np.squeeze(probs_)
        pos = int(np.searchsorted(np.cumsum(probs_), np.random.rand() * np.sum(probs_)))
        c = id2char[pos]
    return gen[1:]


def generate_with_head(head):
    """
    生成一首藏头诗
    """
    states_ = sess.run(initial_state)
    gen = ''
    c = '['
    i = 0
    while c != ']':
        gen += c
        x = np.zeros((BATCH_SIZE, 1))
        x[:, 0] = char2id[c]
        probs_, states_ = sess.run([probs, last_states], feed_dict={X: x, initial_state: states_})
        probs_ = np.squeeze(probs_)
        pos = int(np.searchsorted(np.cumsum(probs_), np.random.rand() * np.sum(probs_)))
        if (c == '[' or c == '。' or c == '，') and i < len(head):
            c = head[i]
            i += 1
        else:
            c = id2char[pos]
    return gen[1:]


print(generate())
print(generate_with_head('书生意气'))
