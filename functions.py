import tensorflow as tf
import numpy as np

class function:

    def __init__(self):
        self

    def test(self):
        print("class test!!!!!!!!")

    def cnn_classifier(x):
        print("NO SKIP CONNECTION")
        # first_layer 32x32x3
        W1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=5e-2))
        L1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


        print(np.shape(L1))
        # second layer 16x16x64
        W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # third layer 8x8x64
        W3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)


        # 4th layer 8x8x128
        W4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))
        L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
        L4 = tf.nn.relu(L4)

        # 5th layer 8x8x256
        W5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2))
        L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
        L5 = tf.nn.relu(L5)

        #6th layer 8x8x256
        W6 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 128], stddev=5e-2))
        L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
        L6 = tf.nn.relu(L6)

        W7 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
        L7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME')
        L7 = tf.nn.relu(L7)

        #fully connected 8x8x128
        fc = tf.reshape(L6, [-1, 8 * 8 * 128])
        W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 1000], stddev=5e-2))
        b_fc1 = tf.Variable(tf.random_normal([1000]))
        L_fc1 = tf.nn.relu(tf.matmul(fc, W_fc1) + b_fc1)

        W_fc2 = tf.Variable(tf.truncated_normal(shape=[1000, 10], stddev=5e-2))
        b_fc2 = tf.Variable(tf.random_normal([10]))
        logit = tf.matmul(L_fc1, W_fc2) + b_fc2
        prediction = tf.nn.softmax(logit)
        print(logit[1])

        return prediction, logit

    def cnn_classifier_with_skip_connection(x):
        print("SKIP CONNECTION ACTIVATED~~~~~~~")
        # first_layer 32x32x3
        W1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=5e-2))
        L1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


        print(np.shape(L1))
        # second layer 16x16x64
        W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # third layer 8x8x64
        W3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)

        # 4th layer 8x8x128
        W4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))
        L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
        L4 = tf.nn.relu(L4)

        # 5th layer 8x8x256
        W5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2))
        L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
        L5 = tf.nn.relu(L5+L4)


        #6th layer 8x8x256
        W6 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 128], stddev=5e-2))
        L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
        L6 = tf.nn.relu(L6+L3)

        W7 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
        L7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME')
        L7 = tf.nn.relu(L7+L3)

        #fully connected 8x8x128
        fc = tf.reshape(L6, [-1, 8 * 8 * 128])
        W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 1000], stddev=5e-2))
        b_fc1 = tf.Variable(tf.random_normal([1000]))
        L_fc1 = tf.nn.relu(tf.matmul(fc, W_fc1) + b_fc1)


        W_fc2 = tf.Variable(tf.truncated_normal(shape=[1000, 100], stddev=5e-2))
        b_fc2 = tf.Variable(tf.random_normal([100]))
        logit = tf.matmul(L_fc1, W_fc2) + b_fc2
        prediction = tf.nn.softmax(logit)

        return prediction, logit