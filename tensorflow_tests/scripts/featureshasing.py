import re
import nltk
from nltk.util import ngrams
from nltk.tag.util import tuple2str, str2tuple
import logging
logger = logging.getLogger("root")
nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')

class FeatureHasher:
    def __init__(self, use_pos_tag=True, use_single_words=True):
        self.use_single_words = use_single_words
        self.use_pos_tag = use_pos_tag
        self.lexicon_words = None
        self.file_name = "subjclueslen1-HLTEMNLP05.tff"
        self.not_re = re.compile("([^\w]|^)(no(t)?)([^\w]|$)", re.VERBOSE | re.I | re.UNICODE)
        self.nt_re = re.compile("([\w]+n't)([^\w]|$)", re.VERBOSE | re.I | re.UNICODE)

    def features_to_words(self, features):
        spliter_re = re.compile("\s")
        words = set([word for feature in features for word in spliter_re.split(feature)])
        if self.use_pos_tag:
            words = map(lambda word: str2tuple(word)[0], words)
        words = filter(lambda word: word, words)
        return words

    def get_lexicon_features_list(self):
        return ["positive_words_count", "negative_words_count"]

    def craete_lexicon_features(self, words):
        if not self.lexicon_words:
            self.lexicon_words = self._read()
        positive = 0
        negative = 0

        for word in words:
            sentiment = self.lexicon_words.get(word, None)
            if sentiment == "positive":
                positive += 1
            elif sentiment == "negative":
                negative += 1

        return positive, negative

    def pos_tagging(self, words):
        if self.use_pos_tag:
            words = map(tuple2str, nltk.pos_tag(words))
        return words

    def bi_gramm(self, words):
        res = []
        idx = 1
        nots_idx = filter(lambda i: bool(self.nt_re.search(words[i])), range(0, len(words)))
        for i in nots_idx: words.insert(i + 1, "not")
        if len(words) == 1: return words[:]
        while idx < len(words):
            previ = self.not_re.search(words[idx - 1])
            curri = self.not_re.search(words[idx])
            nexti = ((idx + 1) < len(words)) and self.not_re.search(words[idx + 1])
            has_not_near = previ or curri or nexti
            if has_not_near and (idx + 1) < len(words):
                res.append(words[idx - 1] + " " + words[idx] + " " + words[idx + 1])

                if previ: idx += 1
            else:
                res.append(words[idx - 1] + " " + words[idx])
            idx += 1
        return res

    def create(self, words):
        single = self.pos_tagging(words)
        bigramms = self.bi_gramm(single)
        tokens = bigramms if not self.use_single_words else (single + bigramms)
        return tokens

    def _read(self):
        with open(self.file_name, 'r') as file:
            lines = file.readlines()
            words = {}

            word_idx = 2
            sentiment_idx = 5
            lable_value_idx = 1

            idx = 0
            for line in lines:
                word_lables = re.compile("\s").split(line)
                word = word_lables[word_idx].split("=")[lable_value_idx]
                idx += 1
                sentiment = word_lables[sentiment_idx].split("=")[lable_value_idx]
                words[word] = sentiment

        return words


