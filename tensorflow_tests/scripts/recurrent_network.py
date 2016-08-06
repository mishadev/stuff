import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from dataproviders import TwetterDataProvider

trainset_provider = TwetterDataProvider(amount=0.5)
trainset_provider.fetch_data()

testset_provider = TwetterDataProvider(words_list=trainset_provider.words_list, train=False, amount=0.1)
testset_provider.fetch_data()

class RecurrentNN:
    def __init__(self, config):
        self.learning_rate = config.get("learning_rate", 0.001)
        self.training_iters = config.get("training_iters", 1000000)
        self.display_step = config.get("display_step", 10)
        self.batch_size = config.get("batch_size", 128)

        self.n_inputs = config["n_input"]
        self.n_steps = config["n_steps"]
        self.n_layers = config["n_layers"]
        self.n_hidden = config["n_hidden"]
        self.n_classes = config["n_classes"]

    def traine(self, ):
        self.n_inputs = config.get("n_input", )

learning_rate = 0.001
training_iters = 1000000
display_step = 10
batch_size = 128

# Network Parameters
n_input = len(trainset_provider.words_list)
n_steps = 1 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 2
n_layers = 1

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    multi_cell = rnn_cell.MultiRNNCell([cell] * n_layers)

    outputs, states = rnn.rnn(multi_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = trainset_provider.next(batch_size)
        batch_x = np.reshape(batch_x, (batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
            if acc > 0.99 and loss < 0.05:
                break

        step += 1
    print "Optimization Finished!"

    test_len = 128
    acc = []
    for idx in range(0, int(len(testset_provider.data) / test_len)):
        test_data, test_label = testset_provider.next(test_len)
        test_data = np.reshape(test_data, (test_len, n_steps, n_input))
        acc.append(sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
        print "Testing Accuracy:", acc[len(acc) - 1]

    print "Avg Accuracy:", np.mean(acc)

