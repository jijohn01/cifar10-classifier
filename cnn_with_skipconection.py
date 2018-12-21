# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.datasets.cifar10 import load_data
import functions

# -------------main-----------
func = functions.function
# func.test()
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# Load CIFAR-10 dataset 50000장
(x_train, y_train), (x_test, y_test) = load_data()
y_train = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

prediction, logit = func.cnn_classifier_with_skip_connection(func, x, keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    loss_val_list = []
    acc_list = []
    acc_val_list = []
    index_list = []

    for i in range(4000):
        # batch size 128
        batch = func.next_batch(func, 128, x_train, y_train.eval())

        if i % 10 == 0:
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1})
            valid_batch = func.next_batch(func, 100, x_test, y_test.eval())
            loss_val = loss.eval(feed_dict={x: valid_batch[0], y: valid_batch[1], keep_prob: 1})
            acc_val = accuracy.eval(feed_dict={x: valid_batch[0], y: valid_batch[1], keep_prob: 1})
            acc = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1})
            if i > 2:
                loss_list.append(loss_print)
                loss_val_list.append(loss_val)
                acc_list.append(acc)
                acc_val_list.append(acc_val)
                index_list.append(i)

            # print(acc)
            print("스텝수 : %d , loss : %f, 정확도(퍼센트) : %f " % (i, loss_print, acc))
            print("오버피팅 확인용 정확도 : %f" % acc_val)

        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})

    print("Learning finished!")

    test_accuracy = 0.0
    for i in range(10):
        test_batch = func.next_batch(func, 1000, x_test, y_test.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1})
        # test 결과 이미지로 확인 부
        #img = test_batch[0][1]
        #label = np.reshape(test_batch[1][1], [1, 10])
        #label = tf.argmax(label, 1).eval()
        #test_predic = logit.eval(
        #    feed_dict={x: np.reshape(test_batch[0][1], [1, 32, 32, 3]), y: np.reshape(test_batch[1][1], [1, 10]),
        #               keep_prob: 1})
        #test_predic = tf.argmax(test_predic, 1).eval()
        #plt.title('prediction : [%d] , label : [%d]' % (test_predic, label))
        #plt.imshow(img)
        #plt.show()
    test_accuracy = test_accuracy / 10;
    print("테스트 데이터 정확도: %f" % test_accuracy)

    # acc 출력하려면 여기서 부터 주석제거
    plt.xlabel('iteration')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.text(3000,test_accuracy+0.1,'Testset error%f'%test_accuracy)
    plt.plot(index_list,acc_list,index_list,acc_val_list,'r-')
    plt.show()
    # loss 출력하려면 여기서 부터 주석제거
    #plt.xlabel('iteration')
    #plt.ylabel('Loss')
    #plt.grid(True)
    #plt.plot(index_list, loss_list, index_list, loss_val_list, 'r-')
    #plt.show()


