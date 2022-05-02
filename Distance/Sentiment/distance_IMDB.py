import tensorflow.compat.v1 as tf
import numpy as np
import re

import Utils
import constants
from Utils import printD

class IMDB:

    def __init__(self):
        self.train_file = './Baseline/Sentiment/data/imdb.train'
        self.dev_file = './Baseline/Sentiment/data/imdb.dev'
        self.test_file = './Baseline/Sentiment/data/test.csv'
        self.ood_cr_file = './Baseline/Sentiment/data/CR.train'
        self.ood_mr_file = './Baseline/Sentiment/data/MR.train'
        self.max_example_len = 400
        self.batch_size = 32
        self.embedding_dims = 50
        self.vocab_size = 5000
        self.num_epochs = 15

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

        if filename != './Baseline/Categorization/data/test.csv':
            x, y = [], []
            with open(filename, "r") as f:
                for line in f:
                    line = re.sub(r'\W+', ' ', line).strip().lower()  # perhaps don't make words lowercase?
                    x.append(line[:-1])
                    x[-1] = ' '.join(word for word in x[-1].split() if word not in stop_words)
                    y.append(line[-1])
            return x, np.array(y, dtype=int)
        else:
            x, y = [], []
            with open(filename, "r") as f:
                for line in f:
                    line = re.sub(',', ' ', line)
                    line = re.sub(r'\W+', ' ', line).strip().lower()  # perhaps don't make words lowercase?
                    x.append(line[:-1])
                    x[-1] = ' '.join(word for word in x[-1].split() if word not in stop_words)
                    y.append(line[-1])
            return x, np.array(y, dtype=int)

    def get_data(self):
        printD('Loading IMDB Data')
        X_train, Y_train = self.load_data(self.train_file)
        X_dev, Y_dev = self.load_data(self.dev_file)
        X_test, Y_test = self.load_data(self.test_file)

        vocab = Utils.get_vocab(X_train)
        X_train = Utils.text_to_rank(X_train, vocab, 5000)
        X_dev = Utils.text_to_rank(X_dev, vocab, 5000)
        X_test = Utils.text_to_rank(X_test, vocab, 5000)

        X_train = Utils.pad_sequences(X_train, maxlen=self.max_example_len)
        X_dev = Utils.pad_sequences(X_dev, maxlen=self.max_example_len)
        X_test = Utils.pad_sequences(X_test, maxlen=self.max_example_len)

        cr_data, cr_labels = self.load_data(self.ood_cr_file)
        cr_data = Utils.text_to_rank(cr_data, vocab, 5000)
        cr_data = Utils.pad_sequences(cr_data, maxlen=self.max_example_len)

        rt_data, rt_labels = self.load_data(self.ood_mr_file)
        rt_data = Utils.text_to_rank(rt_data, vocab, 5000)
        rt_data = Utils.pad_sequences(rt_data, maxlen=self.max_example_len)

        printD('IMDB Data loaded')
        return X_train, Y_train, X_dev, Y_dev, X_test, Y_test, cr_data, cr_labels, rt_data, rt_labels

    def train_model(self):

        X_train, Y_train, X_dev, Y_dev, X_test, Y_test, cr_data, cr_labels, rt_data, rt_labels = self.get_data()
        num_examples = Y_train.shape[0]
        num_batches = num_examples // self.batch_size

        graph = tf.Graph()

        with graph.as_default():
            x = tf.placeholder(dtype=tf.int32, shape=[None, self.max_example_len])
            y = tf.placeholder(dtype=tf.int64, shape=[None])
            is_training = tf.placeholder(tf.bool)

            # add one to vocab size for the padding symbol
            W_embedding = tf.Variable(tf.nn.l2_normalize(
                tf.random_normal([self.vocab_size + 1, self.embedding_dims]), 0), trainable=True)

            w_vecs = tf.nn.embedding_lookup(W_embedding, x)
            pooled = tf.reduce_mean(w_vecs, reduction_indices=[1])

            W_out = tf.Variable(tf.nn.l2_normalize(tf.random_normal([self.embedding_dims, 2]), 0))
            b_out = tf.Variable(tf.zeros([1]))

            logits = tf.matmul(pooled, W_out) + b_out

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
            acc = 100 * tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, 1), y)))

            s = tf.nn.softmax(logits)
            kl_all = tf.log(2.) + tf.reduce_sum(s * tf.log(tf.abs(s) + 1e-10),
                                                                           reduction_indices=[1], keep_dims=True)

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
                np.random.shuffle(indices)
                X_train = X_train[indices]
                Y_train = Y_train[indices]

                for i in range(num_batches):
                    offset = i * self.batch_size

                    x_batch = X_train[offset:offset + self.batch_size]
                    y_batch = Y_train[offset:offset + self.batch_size]

                    _, l, batch_acc = sess.run([optimizer, loss, acc], feed_dict={x: x_batch, y: y_batch})

                    if i % 100 == 0:
                        curr_dev_acc = sess.run(
                            acc, feed_dict={x: X_dev, y: Y_dev})
                        if best_acc < curr_dev_acc:
                            best_acc = curr_dev_acc
                            saver.save(sess, './Baseline/Sentiment/data/best_imdb_model.ckpt')

                print('Epoch %d | Minibatch loss %.3f | Minibatch accuracy %.3f | Dev accuracy %.3f' %
                      (epoch + 1, l, batch_acc, curr_dev_acc))

        # restore variables from disk
        saver.restore(sess, "./Baseline/Sentiment/data/best_imdb_model.ckpt")
        print("Best model restored!")

        print('Dev accuracy:', sess.run(acc, feed_dict={x: X_dev, y: Y_dev}))

        kl_a = sess.run([kl_all], feed_dict={x: X_test, y: Y_test, is_training: False})
        kl_oos1 = sess.run([kl_all], feed_dict={x: cr_data, is_training: False})
        kl_oos2 = sess.run([kl_all], feed_dict={x: rt_data, is_training: False})

        return sess, saver, graph, pooled, logits, x, y, is_training, kl_a[0], kl_oos1[0], kl_oos2[0]



