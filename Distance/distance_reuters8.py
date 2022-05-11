import os

import tensorflow.compat.v1 as tf
import numpy as np
import re

import Utils
import constants
from Utils import printD


class Reuters8:

    def __init__(self):
        self.train_file = './Baseline/Categorization/data/r8-train.txt'
        self.test_file = './Baseline/Categorization/data/r8-test.txt'
        self.batch_size = 32
        self.vocab_size = 1000
        self.num_epochs = 5
        self.n_hidden = 512
        self.nclasses_to_exclude = 2  # 0-3
        np.random.seed(0)
        random_classes = np.arange(8)
        np.random.shuffle(random_classes)
        self.to_include = list(random_classes[:8 - self.nclasses_to_exclude])
        self.to_exclude = list(random_classes[8 - self.nclasses_to_exclude:])

    @staticmethod
    def load_data(filename):
        '''
        :param filename: the system location of the data to load
        :return: the text (x) and its label (y)
                 the text is a list of words and is not processed
        '''

        # stop words taken from nltk
        stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                      'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                      'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                      'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                      'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                      'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                      'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                      'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                      'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn',
                      'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan',
                      'shouldn', 'wasn', 'weren', 'won', 'wouldn']

        x, y = [], []
        with open(filename, "r") as f:
            for line in f:
                line = re.sub(r'\W+', ' ', line).strip()
                x.append(line[1:])
                x[-1] = ' '.join(word for word in x[-1].split() if word not in stop_words)
                y.append(line[0])
        return x, np.array(y, dtype=int)


    def get_data(self):
        printD('Loading Reuters8 Data')
        X_train, Y_train = self.load_data(self.train_file)
        X_test, Y_test = self.load_data(self.test_file)

        vocab = Utils.get_vocab(X_train)
        X_train = Utils.text_to_matrix(X_train, vocab, self.vocab_size)
        X_test = Utils.text_to_matrix(X_test, vocab, self.vocab_size)

        # shuffle
        indices = np.arange(X_train.shape[0])
        np.random.seed(0)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        indices = np.arange(X_test.shape[0])
        np.random.seed(0)
        np.random.shuffle(indices)
        X_test = X_test[indices]
        Y_test = Y_test[indices]

        # split into train/dev
        X_dev = X_train[-500:]
        Y_dev = Y_train[-500:]
        X_train = X_train[:-500]
        Y_train = Y_train[:-500]

        in_sample_examples, in_sample_labels, oos_examples, oos_labels = \
            Utils.partition_data_in_two(X_train, Y_train, self.to_include, self.to_exclude)
        dev_in_sample_examples, dev_in_sample_labels, dev_oos_examples, dev_oos_labels = \
            Utils.partition_data_in_two(X_dev, Y_dev, self.to_include, self.to_exclude)
        test_in_sample_examples, test_in_sample_labels, test_oos_examples, test_oos_labels = \
            Utils.partition_data_in_two(X_test, Y_test, self.to_include, self.to_exclude)

        # safely assumes there is an example for each in_sample class in both the training and dev class
        in_sample_labels = Utils.relabel_in_sample_labels(in_sample_labels)
        dev_in_sample_labels = Utils.relabel_in_sample_labels(dev_in_sample_labels)
        test_in_sample_labels = Utils.relabel_in_sample_labels(test_in_sample_labels)
    
        # use all ood examples
        oos_examples = np.concatenate((oos_examples, dev_oos_examples, test_oos_examples), axis=0)
        oos_labels = np.concatenate((oos_labels, dev_oos_labels, test_oos_labels), axis=0)

        printD('Reuters8 Data loaded')

        return in_sample_examples, in_sample_labels, dev_in_sample_examples, \
               dev_in_sample_labels, test_in_sample_examples, test_in_sample_labels, oos_examples, oos_labels

    def train_model(self):

        in_sample_examples, in_sample_labels, dev_in_sample_examples, \
        dev_in_sample_labels, test_in_sample_examples, test_in_sample_labels, oos_examples, oos_labels = self.get_data()

        num_examples = in_sample_labels.shape[0]
        num_batches = num_examples // self.batch_size

        graph = tf.Graph()

        with graph.as_default():
            x = tf.placeholder(dtype=tf.float32, shape=[None, self.vocab_size])
            y = tf.placeholder(dtype=tf.int64, shape=[None])
            is_training = tf.placeholder(tf.bool)

            # add one to vocab size for the padding symbol
            # tf.random.set_random_seed(0)
            W_h = tf.Variable(tf.nn.l2_normalize(tf.random_normal([self.vocab_size, self.n_hidden]), 0) / tf.sqrt(1 + 0.45))
            b_h = tf.Variable(tf.zeros([self.n_hidden]))

            def gelu_fast(_x):
                return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))

            h = tf.cond(is_training,
                        lambda: tf.nn.dropout(gelu_fast(tf.matmul(x, W_h) + b_h), 0.5),
                        lambda: gelu_fast(tf.matmul(x, W_h) + b_h))
            # tf.random.set_random_seed(0)
            W_out = tf.Variable(
                tf.nn.l2_normalize(tf.random_normal([self.n_hidden, 8 - self.nclasses_to_exclude]), 0) / tf.sqrt(0.45 + 1))
            b_out = tf.Variable(tf.zeros([8 - self.nclasses_to_exclude]))

            logits = tf.matmul(h, W_out) + b_out

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(1e-3, global_step, 4 * num_batches, 0.1, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

            acc = 100 * tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, 1), y)))

            s = tf.nn.softmax(logits)
            kl_all = tf.log(8. - self.nclasses_to_exclude) + tf.reduce_sum(s * tf.log(tf.abs(s) + 1e-10), reduction_indices=[1], keep_dims=True)

        # initialize
        sess = tf.InteractiveSession(graph=graph)
        tf.initialize_all_variables().run()
        # create saver to train model
        saver = tf.train.Saver(max_to_keep=1)

        if constants.RE_TRAIN:

            best_acc = 0

            for epoch in range(self.num_epochs):
                # shuffle data every epoch
                indices = np.arange(num_examples)
                np.random.seed(0)
                np.random.shuffle(indices)
                in_sample_examples = in_sample_examples[indices]
                in_sample_labels = in_sample_labels[indices]

                for i in range(num_batches):
                    offset = i * self.batch_size

                    x_batch = in_sample_examples[offset:offset + self.batch_size]
                    y_batch = in_sample_labels[offset:offset + self.batch_size]

                    _, l, batch_acc = sess.run([optimizer, loss, acc],
                                               feed_dict={x: x_batch, y: y_batch, is_training: True})

                curr_dev_acc = sess.run(
                    acc, feed_dict={x: dev_in_sample_examples, y: dev_in_sample_labels, is_training: False})
                if best_acc < curr_dev_acc:
                    best_acc = curr_dev_acc
                    saver.save(sess, "Baseline/Categorization/data/best_r8_model.ckpt")

                print('Epoch %d | Minibatch loss %.3f | Minibatch accuracy %.3f | Dev accuracy %.3f' %
                      (epoch + 1, l, batch_acc, curr_dev_acc))

            # restore variables from disk
            saver.restore(sess, "Baseline/Categorization/data/best_r8_model.ckpt")
            print("Best model restored!")

            print('Dev accuracy:',
                  sess.run(acc, feed_dict={x: dev_in_sample_examples, y: dev_in_sample_labels, is_training: False}))
        else:
            # restore variables from disk
            saver.restore(sess, os.path.join(constants.MODELS_FOLDER,"best_r8_model.ckpt"))
            print("Best model restored without retraining!")
            print('Dev accuracy:', sess.run(acc, feed_dict={x: dev_in_sample_examples, y: dev_in_sample_labels, is_training: False}))

        kl_a = sess.run([kl_all], feed_dict={x: in_sample_examples, y: in_sample_labels,  is_training: False})
        kl_oos = sess.run([kl_all], feed_dict={x: oos_examples,  is_training: False})

        return sess, saver, graph, h, logits, x, y, is_training, kl_a[0], kl_oos[0]
