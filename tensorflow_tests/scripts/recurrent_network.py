import sys
import random
from timeit import default_timer as timer
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from dataproviders import TwetterDataProvider, BatchDataReader
import logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pretty_timer(start, end):
    time = end - start
    return str(int(time / 60)) + "m " + str(int(time % 60)) + "s " + str(int((time % 60) * 1000 % 1000)) + "ms"

start_timer = timer()
batch_reader = BatchDataReader({"trains":0.95, "test":0.05})
data_provider = TwetterDataProvider(amount=1.0, use_single_words=True, use_lexicon_features=True)
batch_reader.use_data(*data_provider.fetch_data())
received_data_timer = timer()

class RecurrentNN:
    def __init__(self, config):
        self.learning_rate = config.get("learning_rate", 0.001)
        self.training_iters = config.get("training_iters", 100000)
        self.display_step = config.get("display_step", 10)
        self.batch_size = config.get("batch_size", 128)

        self.n_inputs = config["n_input"]
        self.n_steps = config["n_steps"]
        self.n_layers = config["n_layers"]
        self.n_hidden = config["n_hidden"]
        self.n_classes = config["n_classes"]


learning_rate = 0.001
training_iters = 1000000
display_step = 10
batch_size = 128

n_input = len(data_provider.columns)
n_steps = 1 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 2
n_layers = 1

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)

    cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    multi_cell = rnn_cell.MultiRNNCell([cell] * n_layers)

    outputs, states = rnn.rnn(multi_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
# saver = tf.train.Saver({"weights": weights, "biases": biases})

def plot(training, testing):
    import matplotlib.pyplot as plt
    colors = plt.cm.rainbow(np.linspace(0, 1, 2))
    plt.plot(range(0, len(training)), training, color='b')
    plt.plot(range(0, len(testing)), testing, color='r')
    file_name = str(random.random())[2:]
    # Also plot centroid
    plt.savefig("./images/" + file_name + ".png")

with tf.Session() as sess:
    sess.run(init)
    step = 1
    test_seq_len = batch_reader.bucket_len("test") / batch_size
    test_accs = np.zeros(shape=(test_seq_len), dtype=np.float)
    traning_acc = []
    testing_acc = []
    while step * batch_size < training_iters:
        batch_x, batch_y = batch_reader.next(batch_size, "trains")
        batch_x = np.reshape(batch_x, (batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            test_data, test_label = batch_reader.next(batch_size, "test")
            test_data = np.reshape(test_data, (batch_size, n_steps, n_input))
            test_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_label})

            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

            test_accs[int(step / display_step) % test_seq_len] = test_acc
            avg_acc = np.mean(test_accs)
            traning_acc.append(acc)
            testing_acc.append(test_acc)

            # logger.info("Iter " + str(step*batch_size) + \
                # ", Minibatch Loss= {:.6f}".format(loss) + \
                # ", Training Accuracy= {:.5f}".format(acc) +\
                # ", Tasting Accuracy= {:.5f}".format(avg_acc))
            if avg_acc > 0.94:
                break

        step += 1
    logger.info("Optimization Finished!")

    finish_timer = timer()
    logger.info("-"*50)
    logger.info("ML :" + pretty_timer(received_data_timer, finish_timer))
    logger.info("Total :" + pretty_timer(start_timer, finish_timer))

    print "=="*40
    print "Testing Accuracy: {:.4f}".format(avg_acc)
    # if avg_acc > 0.94:
        # path = saver.save(sess, "recurent_model.ckpt")
        # print "saved to :" + path
    print "=="*40
    plot(traning_acc, testing_acc)


