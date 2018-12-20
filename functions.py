import tensorflow as tf
import numpy as np

class function:

    def __init__(self):
        self

    def test(self):
        print("class test!!!!!!!!")

    def cnn_classifier(x):
        # first_layer
        W1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], stddev=5e-2))
        L1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        print(np.shape(L1))
        # second layer
        W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=5e-2))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # third layer
        W3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.dropout(L3, 0.5)

        # 4th layer
        W4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
        L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
        L4 = tf.nn.relu(L4)

        # 5th layer
        W5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
        L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
        L5 = tf.nn.relu(L5)

        fc = tf.reshape(L5, [-1, 8 * 8 * 128])
        W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 400], stddev=5e-2))
        b_fc1 = tf.Variable(tf.random_normal([400]))
        L_fc1 = tf.nn.relu(tf.matmul(fc, W_fc1) + b_fc1)
        L_fc1 = tf.nn.dropout(L_fc1, 0.5)

        W_fc2 = tf.Variable(tf.truncated_normal(shape=[400, 10], stddev=5e-2))
        b_fc2 = tf.Variable(tf.random_normal([10]))
        logit = tf.matmul(L_fc1, W_fc2) + b_fc2
        prediction = tf.nn.softmax(logit)
        print(logit[1])

        return prediction, logit