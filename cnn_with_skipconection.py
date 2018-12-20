# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.datasets.cifar100 import load_data
import functions


def next_batch(num, data, labels):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
def cnn_classifier(x):
    #first_layer
    W1 = tf.Variable(tf.truncated_normal(shape=[3,3,3,32], stddev=5e-2))
    L1 = tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    print(np.shape(L1))
    #second layer
    W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=5e-2))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # third layer
    W3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)


    fc = tf.reshape(L3,[-1,8*8*128])
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8*8*128,400], stddev=5e-2))
    b_fc1 = tf.Variable(tf.random_normal([400]))
    L_fc1 = tf.nn.relu(tf.matmul(fc,W_fc1)+b_fc1)
    L_fc1 = tf.nn.dropout(L_fc1, 0.8)

    W_fc2 = tf.Variable(tf.truncated_normal(shape=[400,10], stddev=5e-2))
    b_fc2 = tf.Variable(tf.random_normal([10]))
    logit = tf.matmul(L_fc1,W_fc2)+b_fc2
    prediction = tf.nn.softmax(logit)
    print(logit[1])

    return prediction, logit

#-------------main-----------
func = functions.function
#func.test()
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 100])

# Load CIFAR-10 dataset 50000장
(x_train,y_train),(x_test,y_test) = load_data()
y_train = tf.squeeze(tf.one_hot(y_train,100),axis=1)
y_test = tf.squeeze(tf.one_hot(y_test,100),axis=1)
print(np.shape(y_train))
prediction, logit = func.cnn_classifier_with_skip_connection(x)
#imgplot = plt.imshow(x_train[1])
#plt.show()

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logit))
train_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

correct = tf.equal(tf.argmax(logit,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list=[]

    for i in range(10000):
        #batch size 128
        batch = next_batch(128,x_train,y_train.eval())

        if i%100 ==0:
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1]})
            valid_batch = next_batch(100,x_test,y_test.eval())
            acc_val = accuracy.eval(feed_dict={x: valid_batch[0], y: valid_batch[1]})
            acc = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
            if i > 2:
                loss_list.append(loss_print)
           # print(acc)
            print("스텝수 : %d , loss : %f, 정확도(퍼센트) : %f "%(i, loss_print,acc))
            print("오버피팅 확인용 정확도 : %f"%acc_val)

        sess.run(train_step,feed_dict={x: batch[0],y:batch[1]})

    print("Learning finished!")
    plt.plot(loss_list)
    plt.show()
    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1]})
    test_accuracy = test_accuracy / 10;
    print("테스트 데이터 정확도: %f" % test_accuracy)
