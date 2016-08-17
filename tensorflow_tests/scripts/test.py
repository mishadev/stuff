import re

class TextUtils(object):
    def __init__(self):
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
            new_row = {}
            for column_name in column_names:
                new_row[column_name] = row[column_name] if column_name in row else 0
            matrix.append(new_row)
        return matrix

    def _count_words_many(self, texts):
        word_count_by_text = []
        word_appearance = {}

        for text in texts:
            text_word_count = self._count_words(text)
            word_count_by_text.append(text_word_count)

            for word in text_word_count.keys():
                word_appearance.setdefault(word, 0)
                word_appearance[word] += 1

        return word_appearance, word_count_by_text

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
        return list(filter(None, map(str.lower, words)))

    def _word_appearance_check(self, app_count, texts_count):
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

        return white_list

class TwetterDataProvider(object):
    def __init__(self, first=True, amount=1.0):
        self.batch_id = 0
        self.first = first
        self.amount = amount
        self.file_name = "Sentiment140.tenPercent.sample.tweets.tsv"
        self.text_utils = TextUtils()

    def fetch_data(self):
        texts, labels = self._read()
        self.data = self.text_utils.count_meaningful_words(texts)
        self.labels = labels

    def _read(self):
        with open(self.file_name, 'r') as file:
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
        return batch_data, batch_labels

provider = DataProvider(amount=0.008)
provider.fetch_data()
