'''
A Dynamic Reccurent Neural Network (LSTM) implementation example using
TensorFlow library. This example is using a toy dataset to classify linear
sequences. The generated sequences have variable length.

Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import codecs
import tensorflow as tf
import random

def log(text):
    print ("="*20)
    print (text)
    print ("="*20)


# ====================
#  TOY DATA GENERATOR
# ====================
class TextUtils(object):
    def __init__(self, max_words_amount = 0.0):
        self.max_words_amount = max_words_amount
        self.lower_boundary_count = 2
        self.upper_boundary_probability = 0.6
        self.min_chars = 3

    def count_meaningful_words(self, texts):
        word_appearance, text_word_count = self._count_words_many(texts)
        word_list = self._get_word_list(word_appearance, len(texts))

        return self._create_matrix(text_word_count, word_list)

    def _create_matrix(self, rows, column_names):
        matrix = []
        for row in rows:
            row_dict = {}
            for column_name in column_names:
                row_dict[column_name] = row[column_name] if column_name in row else 0
            matrix.append(row_dict)
        return matrix

    def _count_words_many(self, texts):
        word_count_by_text = []
        word_appearance = {}

        for text in texts:
            text_word_count = self._count_words(text)
            word_count_by_text.append(text_word_count)

            for word, count in text_word_couna.keys():
                word_appearance.setdefault(word, 0)
                word_appearance[word] += 1

        return words_appearance, text_word_count

    def _count_words(self, text):
        words = self._get_words(text)

        result = {}
        for word in words:
            result.setdefault(word, 0)
            result[word] += 1

        return result

    def _get_words(self, text):
        txt = re.compile(r'<[^>]+>').sub('', text)
        words = re.compile(r'[^a-z^a-z]+').split(txt)
        return list(filter(one, map(str.lower, words)))

    def _word_appearance_check(self, texts_count):
        probability = float(app_count) / texts_count
        lower_boundary_probability = self.lower_boundary_count / texts_count
        upper_boundary_probability = self.upper_boundary_probability

        return (probability > lower_boundary_probability
                and probability < upper_boundary_probability)

    def _word_chars_check(self, word):
        return len(word) >= self.min_chars

    def _get_word_list(self, words_appearance, texts_count):
        if texts_count <= 1:
            return list(words_appearance.keys())

        white_list = []
        black_list = []
        for word, app_count in words_appearance.items():
            if self._word_appearance_check(app_count, texts_count) and self._word_chars_check(word):
                white_list.append(word)
            else:
                log("black : " + word + " : " + str(probability) + ", from:" + texts_count)

        return white_list


class NeuralNetwork(object):
    default_config = {
        "learning_rate": 0.01,
        "training_iters" : 1000 * 1000,
        "batch_size" : 128,
        "learn_heardbeat_interval" : 10,
        "feature_max_len" : 20000,
        "n_hidden" : 128,
        "n_classes" : 2
    }

    def get_configuration(self, config):
        cfg = {}
        for name, value in default_config.items():
            cfg[name] = config.get(name, value)
        return cfg

    def __init__(self, config):
        self.__config = self.get_configuration(config)

    def learn(self, texts, lables):
        words_count_by_text, words_appearance

trainset = GetSequenceData(max_lengs=seq_max_len)
testset = GetSequenceData(max_lengs=seq_max_len, trainset=False)

seq_max_len = trainset.max_lengs# Sequence max length
log(seq_max_len)
# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

class DataProvider(object):
    def _read(self):
        with codecs.open(file_name, 'r') as file:
            lines = file.readlines()
            idx = 0
            texts = []
            labels = []
            if self.first:
                lines = lines[0:int(len(lines) * self.amount) - 1]
            else:
                liens = lines[int(len(lines) * self.amount):0]

            for line in lines:
                text_with_lable = line.strip().split('\t')

                texts.append(text_with_lable[0].strip())
                label = int(p[1] == "positive")
                labels.append([label, abs(label - 1)])
        return texts, labels


    def __init__(self, first=True, amount=1.0):
        self.data = []
        self.labels = []
        self.batch_id = 0
        self.first = first
        self.amount = amount
        self.file_name = "Sentiment140.tenPercent.sample.tweets.tsv"
        self.data, self.labels = self._read()

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_iters = 1000000
batch_size = 128
display_step = 10
seq_max_len = 400
# Network Parameters
n_hidden = 128 # hidden layer num of features
n_classes = 2 # linear sequence or not

trainset = GetSequenceData(max_lengs=seq_max_len)
testset = GetSequenceData(max_lengs=seq_max_len, trainset=False)

seq_max_len = trainset.max_lengs# Sequence max length
log(seq_max_len)
# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, 1])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, seq_max_len, x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e, if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

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
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen})
