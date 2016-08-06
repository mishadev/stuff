import numpy as np
import re
import collections

class MeaningfulWordsFilter():
    def __init__(self,
            min_word_length=3,
            min_word_appearence=2,
            max_word_probability=0.6):
        self.min_word_length = min_word_length
        self.min_word_appearence = min_word_appearence
        self.max_word_probability = max_word_probability

    def _word_probability_check(self, appearance_count, texts_count):
        probability = float(appearance_count) / texts_count
        lower_boundary_probability = float(self.min_word_appearence) / texts_count
        upper_boundary_probability = self.max_word_probability

        return ((probability > lower_boundary_probability) and
            (probability < upper_boundary_probability))

    def _word_chars_check(self, word):
        return len(word) >= self.min_word_length

    def get_white_list(self, words_appearance, texts_count):
        if texts_count <= 1:
            return list(words_appearance.keys())

        white_list = []
        for word, word_appearance in words_appearance.items():
            if (self._word_chars_check(word) and
                self._word_probability_check(word_appearance, texts_count)):
                white_list.append(word)
        return white_list

class TextUtils(object):
    def count_words(self, texts, word_list=None):
        word_appearance, text_word_count = self._count_words_many(texts)
        if not word_list:
            words_filter = MeaningfulWordsFilter()
            word_list = words_filter.get_white_list(word_appearance, len(texts))

        return word_list, self._create_matrix(text_word_count, word_list)

    def _create_matrix(self, rows, column_names):
        matrix = np.zeros(shape=(len(rows), len(column_names), 1))
        column_name_ids = dict(zip(column_names, range(0, len(column_names))))
        idx = 0
        for row in rows:
            for column_name, value in row.items():
                if column_name in column_name_ids:
                    matrix[idx][column_name_ids[column_name]]=[value]
            idx += 1
        return matrix

    def _count_words_many(self, texts):
        word_count_by_text = []
        word_appearance = {}

        for text in texts:
            text_word_count = collections.Counter(self._get_words(text))
            word_count_by_text.append(text_word_count)

            for word in text_word_count.keys():
                word_appearance.setdefault(word, 0)
                word_appearance[word] += 1

        return word_appearance, word_count_by_text

    def _get_words(self, text):
        txt = re.compile(r'<[^>]+>').sub('', text)
        words = re.compile(r'[^a-z^a-z]+').split(txt)
        return list(filter(None, map(str.lower, words)))


class TwetterDataProvider(object):
    def __init__(self, words_list=None, train=True, amount=0.8):
        self.batch_id = 0
        self.train = train
        self.amount = amount
        self.words_list = words_list
        self.file_name = "Sentiment140.tenPercent.sample.tweets.tsv"
        self.text_utils = TextUtils()

    def fetch_data(self):
        texts, labels = self._read()
        self.words_list, self.data = self.text_utils.count_words(texts, self.words_list)
        self.labels = labels

    def _read(self):
        with open(self.file_name, 'r') as file:
            lines = file.readlines()
            idx = 0
            texts = []
            labels = []
            if self.train:
                lines = lines[:int(len(lines) * self.amount) - 1]
            else:
                lines = lines[-int(len(lines) * self.amount):]

            for line in lines:
                text_with_lable = line.strip().split('\t')

                texts.append(text_with_lable[0].strip())
                label = int(text_with_lable[1] == "positive")
                labels.append([label, abs(label - 1)])
        return texts, labels

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0

        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))

        data_length = len(batch_data)
        if data_length < batch_size:
            additional_data, additional_labels = self.next(batch_size - data_length)
            batch_data = np.concatenate((batch_data, additional_data), axis=0)
            batch_labels = np.concatenate((batch_labels, additional_labels), axis=0)

        return batch_data, batch_labels
'''
d = TwetterDataProvider(amount=0.01)
d.fetch_data()
print "train"
for idx in range(0, 10):
    x, y = d.next(128)
    print np.shape(x)
    print np.shape(y)

t = TwetterDataProvider(words_list=d.words_list, train=False, amount=0.01)
t.fetch_data()
print "test"
for idx in range(0, 10):
    x, y = t.next(128)
    print np.shape(x)
    print np.shape(y)
'''
