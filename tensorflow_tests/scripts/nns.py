import random
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from dataproviders import BatchDataReader
import logging
logger = logging.getLogger("root")

class RecurrentNN:
    def __init__(self, config):
        self.learning_rate = config.get("learning_rate", 0.001)
        self.accuracy_threshold = config.get("accuracy_threshold", 0.95)
        self.training_iters = config.get("training_iters", 1000 * 1000)
        self.display_step = config.get("display_step", 10)
        self.batch_size = config.get("batch_size", 128)

        self.n_input = config["n_input"]
        self.n_steps = config["n_steps"]
        self.n_layers = config["n_layers"]
        self.n_hidden = config["n_hidden"]
        self.n_classes = config["n_classes"]

        self.traing_data_ratio = config.get("traing_data_ratio", 0.95)
        self.batch_reader = BatchDataReader({
            "trains": self.traing_data_ratio,
            "test": (1 - self.traing_data_ratio)})
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

    def learn(self, features_data, labels_data):
        self.batch_reader.use_data(features_data, labels_data)

        features = tf.placeholder("float", [None, self.n_steps, self.n_input])
        labels = tf.placeholder("float", [None, self.n_classes])

        prediction = self._inner_predict(features)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            batch_size = self.batch_size
            test_batches_count = self.batch_reader.bucket_len("test") / batch_size
            test_accs = np.zeros(shape=(test_batches_count))
            traning_acc = []
            testing_acc = []
            while step * batch_size < self.training_iters:
                batch_data, batch_labels = self.batch_reader.next(batch_size, "trains")
                batch_data = np.reshape(batch_data, (batch_size, self.n_steps, self.n_input))
                sess.run(optimizer, feed_dict={features: batch_data, labels: batch_labels})
                if step % self.display_step == 0:
                    test_data, test_label = self.batch_reader.next(batch_size, "test")
                    test_data = np.reshape(test_data, (batch_size, self.n_steps, self.n_input))
                    test_acc = sess.run(
                            accuracy,
                            feed_dict={
                                features: test_data,
                                labels: test_label})

                    acc = sess.run(
                            accuracy,
                            feed_dict={
                                features: batch_data,
                                labels: batch_labels})
                    loss = sess.run(
                            cost,
                            feed_dict={
                                features: batch_data,
                                labels: batch_labels})

                    test_accs[int(step / self.display_step) % test_batches_count] = test_acc
                    traning_acc.append(acc)
                    testing_acc.append(test_acc)
                    avg_acc = np.mean(test_accs)

                    logger.info("Iter " + str(step*batch_size) + \
                        ", Minibatch Loss= {:.6f}".format(loss) + \
                        ", Training Accuracy= {:.5f}".format(acc) +\
                        ", Tasting Accuracy= {:.5f}".format(avg_acc))
                    if avg_acc > self.accuracy_threshold:
                        break

                step += 1
        self.plot(traning_acc, testing_acc)
        logger.info("Accuracy Threshold: " + str(self.accuracy_threshold))
        logger.info("Accuracy: " + str(avg_acc))
        logger.info("Optimization Finished!")

    def predict(self, features):
        with tf.Session() as sess:
            self._inner_predict(features)

    def _inner_predict(self, features):
        features = tf.transpose(features, [1, 0, 2])
        features = tf.reshape(features, [-1, self.n_input])
        features = tf.split(0, self.n_steps, features)

        cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        multi_cell = rnn_cell.MultiRNNCell([cell] * self.n_layers)

        outputs, states = rnn.rnn(multi_cell, features, dtype=tf.float32)

        return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

    def plot(self, training, testing):
        import matplotlib.pyplot as plt
        colors = plt.cm.rainbow(np.linspace(0, 1, 2))
        plt.plot(range(0, len(training)), training, color='b')
        plt.plot(range(0, len(testing)), testing, color='r')
        file_name = str(random.random())[2:]
        # Also plot centroid
        plt.savefig("./images/" + file_name + ".png")

