'''
A Dynamic Reccurent Neural Network (LSTM) implementation example using
TensorFlow library. This example is using a toy dataset to classify linear
sequences. The generated sequences have variable length.

Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import random

def log(text):
    print ("="*20)
    print (text)
    print ("="*20)


# ====================
#  TOY DATA GENERATOR
# ====================
class GerFromTwetter(object):
    def __init__(self, max_lengs=0):
        self.max_lengs=max_lengs
        self.data = []
        self.labels = []
        self.seqlen = []

    def get_word_list(words_appearance, tweet_count):
        if tweet_count <= 1:
            return list(words_appearance.keys())

        white_list = []
        black_list = []
        for word, app_count in words_appearance.items():
            probability = float(app_count) / tweet_count
            one_app = 1 / tweet_count
            if probability > one_app and probability < 0.7:
                white_list.append(word)
            else:
                log("black : " + word + " : " + str(probability))

        return white_list

    def count_words(tweets):
        tweets_words_count = []
        words_appearance = {}

        for tweet in tweets:
            words_count = self.get_word_count(tweet)
            tweets_words_count.append(words_count)
            for word_count in words_count:
                words_appearance.setdefault(word_count.word, 0)
                words_appearance[word_count.word] += 1
        return words_appearance, tweets_words_count

        word_list = get_word_list(words_appearance, len(tweets))
        write_to_file("words.txt", word_list, tweets_words_count)

    def write_to_file(file_name, word_list, tweets_words_count):
        log("write file")
        log("words : %s" % len(word_list))
        log("tweets : %s" % len(tweets_words_count))

        with open(file_name,'w') as file:
            file.write('tweets')
            for word in word_list:
                file.write('\t%s' % word)
                file.write('\n')

                for word_count in tweets_words_count:
                    idx += 1
                    file.write(idx)
                    if word in word_count:
                        file.write('\t%d' % word_count[word])
                    else:
                        file.write('\t0')

                file.write('\n')

class TextUtils(object):
    def __init__(self, config):
        self.upper_boundary_probability = 0.7
        self.lower_boundary_count = 1

    def count_words(self, texts):
        words_count_by_text = []
        words_appearance = {}

        for text in texts:
            words_count = self.get_words_count(text)
            words_count_by_text.append(words_count)

            for word_count in words_count:
                words_appearance.setdefault(word_count.word, 0)
                words_appearance[word_count.word] += 1
        return words_appearance, tweets_words_count

        word_list = get_word_list(words_appearance, len(tweets))
        write_to_file("words.txt", word_list, tweets_words_count)

    def get_words(self, text):
        txt = re.compile(r'<[^>]+>').sub('', text)
        words = re.compile(r'[^a-z^a-z]+').split(txt)
        return list(filter(one, map(str.lower, words)))

    def get_words_count(self, text):
        words = self.get_words(text)

        result = {}
        for word in words:
            result.setdefault(word, 0)
            result[word] += 1

        return result

    def get_word_list(self, words_appearance, tweet_count):
        if tweet_count <= 1:
            return list(words_appearance.keys())

        white_list = []
        black_list = []
        for word, app_count in words_appearance.items():
            probability = float(app_count) / tweet_count
            one_app = 1 / tweet_count
            if probability > one_app and probability < 0.7:
                white_list.append(word)
            else:
                log("black : " + word + " : " + str(probability))

        return white_list


    def write_to_file(file_name, word_list, tweets_words_count):
        log("write file")
        log("words : %s" % len(word_list))
        log("tweets : %s" % len(tweets_words_count))

        with open(file_name,'w') as file:
            file.write('tweets')
            for word in word_list:
                file.write('\t%s' % word)
                file.write('\n')

                for word_count in tweets_words_count:
                    idx += 1
                    file.write(idx)
                    if word in word_count:
                        file.write('\t%d' % word_count[word])
                    else:
                        file.write('\t0')

                file.write('\n')



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

    def predict(self, text):


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



class GetSequenceData(object):
    def read_data(self, file_name):
        with open(file_name, 'r') as file:
            tweets = file.readlines()
            idx = 0
            data = []
            labels = []
            test_tweets = tweets[:int(len(lines) * 0.7)]
            train_tweets = tweets[int(len(lines) * 0.7) + 1:]
            for tweet in train_tweets if self.trainset else test_tweets:
                tweet_with_lable = tweet.strip().split('\t')


                data.append(map(lambda x: [ord(x)], p[0]))
                label = int(p[1] == "positive")
                labels.append([label, abs(label - 1)])
        return data, labels


    def __init__(self, trainset=True, max_lengs=0):
        self.batch_id = 0
        self.trainset = trainset
        self.data = []
        self.labels = []
        self.seqlen = []
        self.max_lengs = max_lengs
        tweets, labels = self.read_data("Sentiment140.tenPercent.sample.tweets.tsv")
        tweets_count = len(tweets)
        for idx in range(0, tweets_count):
            tweet = tweets[idx]
            seqlen = len(tweet)
            self.seqlen.append(seqlen)
            line += [[0.] for i in range(self.max_lengs - seqlen)]
            lable = labels[idx]
            self.data.append(line)
            self.labels.append(lable)

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


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
