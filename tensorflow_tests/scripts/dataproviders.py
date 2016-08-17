import codecs
from multiprocessing import Pool
import numpy as np
import re
import os.path
import math
import collections
import nltk
from tweepy import OAuthHandler
from tokenizers import TweetTokenizerPlus
from featureshasing import FeatureHasher
nltk.download('punkt')
import logging
logger = logging.getLogger("root")


class FeatureFilter():
    def __init__(self, threshold=0.4, min_appearance=1):
        self.threshold = threshold
        self.min_appearance = min_appearance

    def _calculate_appearance(self, features_by_text, labels):
        appearances = {}
        for idx in range(0, len(features_by_text)):
            for feature in features_by_text[idx]:
                appearances.setdefault(feature, [0, 0])
                # calculate positive and negative appearances
                appearances[feature] = np.sum([appearances[feature], labels[idx]], axis=0)
        return appearances

    def _calculate_probability_metric(self, features_by_text, labels):
        metrics = {}
        probability = {}
        total = np.sum(labels, axis=0)
        appearances = self._calculate_appearance(features_by_text, labels)
        for feature, appearance in appearances.items():
            Ppos = float(appearance[0])/total[0]
            Pneg = float(appearance[1])/total[1]
            metrics[feature] = 1.0/2 * (1 - min(Ppos, Pneg)/max(Ppos, Pneg))
            probability[feature] = float(np.sum(appearance))/len(labels)

        return metrics, probability

    def filter_features(self, features_by_text, labels):
        feature_metric, feature_prob = self._calculate_probability_metric(features_by_text, labels)
        features = []
        exclude_sln_count = 0
        exclude_prob_count = 0
        exclude_both = 0
        min_prob = float(self.min_appearance) / len(labels)
        for feature in feature_metric.keys():
            include_sln = feature_metric[feature] > self.threshold
            include_prob = feature_prob[feature] > min_prob
            if include_sln and include_prob:
                features.append(feature)
            else:
                if include_sln and not include_prob: exclude_prob_count += 1
                elif include_prob and not include_sln: exclude_sln_count += 1
                else: exclude_both += 1
        logger.info(" filter(sln) exclude features: " + str(exclude_sln_count))
        logger.info(" filter(prob) exclude features: " + str(exclude_prob_count))
        logger.info(" filter include features: " + str(len(features)))
        return features

class DatasetBuilder(object):
    def __init__(self, use_pos_tag=False, use_single_words=True, use_lexicon_features=True):
        self.use_lexicon_features = use_lexicon_features
        self.feature_filter = FeatureFilter()
        self.tokenizer = TweetTokenizerPlus(preserve_case=True,
                strip_handles=True, reduce_len=True)
        self.hasher = FeatureHasher(use_pos_tag=use_pos_tag, use_single_words=use_single_words)

    def build_dataset(self, features_list, texts):
        logger.info(" start getting features")
        features_by_text = self._get_texts_features(texts)
        logger.info(" start filtering tweets")
        features_matrix, texts_to_remove = self._create_features(
                features_list, features_by_text)

        if self.use_lexicon_features:
            logger.info(" start calculating lexicon feature")
            lexicon_features_list, lexicon_features_matrix = self._calculate_lexicon_feature(
                    features_by_text, texts_to_remove)
            features_list, features_matrix = self._append(
                    features_list, features_matrix,
                    lexicon_features_list, lexicon_features_matrix)

        logger.info(" total feature: " + str(len(features_list)))
        logger.info(" total samples: " + str(len(features_matrix)))
        return features_list, feature_matrix, texts_to_remove

    def build_labeled_dataset(self, texts, labels):
        logger.info(" start getting features")
        labels_list, labels_matrix = self._create_labels_matrix(labels)
        features_by_text = self._get_texts_features(texts)
        logger.info(" start filtering features")
        features_list = self.feature_filter.filter_features(features_by_text, labels_matrix)
        features_matrix, features_less_text_ids = self._create_features(
                features_list, features_by_text)
        labels_matrix = np.delete(labels_matrix, features_less_text_ids, axis=0)

        if self.use_lexicon_features:
            logger.info(" start calculating lexicon feature")
            lexicon_features_list, lexicon_features_matrix = self._calculate_lexicon_feature(
                    features_by_text, features_less_text_ids)
            features_list, features_matrix = self._append(
                    features_list, features_matrix,
                    lexicon_features_list, lexicon_features_matrix)

        logger.info(" total feature: " + str(len(features_list)))
        logger.info(" total samples: " + str(len(features_matrix)))
        return features_list, features_matrix, labels_list, labels_matrix

    def _append(self,
            f_features_list, f_features_matrix,
            s_features_list, s_features_matrix):

        features_list = np.append(f_features_list, s_features_list, axis=0)
        features_matrix = np.append(f_features_matrix, s_features_matrix, axis=1)

        return features_list, features_matrix

    def _create_labels_matrix(self, labels):
        return ["positive", "negative"], map(lambda label: [label, abs(label - 1)], labels)

    def _calculate_lexicon_feature(self, features_by_text, features_less_text_ids):
        lexicon_features_matrix = np.zeros(shape=(
                len(features_by_text) - len(features_less_text_ids),
                2
            ),
            dtype=np.uint8)
        idx = 0
        lexicon_features_count = 0
        lexicon_features_list = self.hasher.get_lexicon_features_list()
        for text_id in range(0, len(features_by_text)):
            if text_id not in features_less_text_ids:
                features = features_by_text[text_id]
                words = self.hasher.features_to_words(features)
                pos, neg = self.hasher.craete_lexicon_features(words)
                lexicon_features_matrix[idx][0] = pos
                lexicon_features_matrix[idx][1] = neg
                lexicon_features_count += min(1, max(pos, neg))
                idx += 1

        logger.info(" useful lexicon features: " + str(lexicon_features_count) + "/" + str(len(lexicon_features_matrix)))
        return lexicon_features_list, lexicon_features_matrix

    def _create_features(self, features_list, features_by_text):
        feature_matrix = np.zeros(shape=(
                len(features_by_text),
                len(features_list)
            ),
            dtype=np.uint8)
        features_list_ids = dict(zip(features_list, range(0, len(features_list))))
        feature_less_text_ids = []
        texts_len = len(features_by_text)
        for text_idx in range(0, texts_len):
            has_values = False
            for features in features_by_text[text_idx]:
                if features in features_list_ids:
                    has_values = True
                    feature_matrix[text_idx][features_list_ids[features]]=1
            if not has_values: feature_less_text_ids.append(text_idx)
        feature_matrix = np.delete(feature_matrix, feature_less_text_ids, axis=0)
        logger.info(" feature less tweets: " + str(len(feature_less_text_ids)) + "/" + str(texts_len))
        logger.info(" featured tweets: " + str(len(feature_matrix)) + "/" + str(texts_len))
        return feature_matrix, feature_less_text_ids

    # hack to make Pool() work
    def __call__(self, text):
        return self._get_text_features(text)

    def _get_text_features(self, text):
        tokens = self.tokenizer.tokenize(text)
        features = self.hasher.create(tokens)
        return features

    def _get_texts_features(self, texts):
        pool = Pool()
        features_by_text = pool.map(self, texts)
        pool.terminate()
        return features_by_text


class BatchDataReader(object):
    def __init__(self, buckets={"def":1}):
       self.buckets = buckets

    def use_data(self, data, labels):
        self.data = data
        self.labels = labels
        self.batch_ids = {}
        self.batch_boundary = {}
        offset = 0
        for name, ratio in self.buckets.items():
            next_offset = offset + int(len(self.data) * ratio)
            self.batch_ids[name] = offset
            self.batch_boundary[name] = (offset, next_offset - 1)
            offset = next_offset

    def bucket_len(self, name="def"):
        if not self.batch_ids or name not in self.batch_ids: return

        lower, upper = self.batch_boundary[name]
        return upper - lower

    def next(self, size, name="def"):
        if not self.batch_ids or name not in self.batch_ids: return

        lower, upper = self.batch_boundary[name]
        if self.batch_ids[name] == upper:
            self.batch_ids[name] = lower

        batch_end_id = min(self.batch_ids[name] + size, upper)
        batch_data = self.data[self.batch_ids[name]:batch_end_id]
        batch_labels = self.labels[self.batch_ids[name]:batch_end_id]
        self.batch_ids[name] = batch_end_id

        data_length = len(batch_data)
        if data_length < size:
            additional_data, additional_labels = self.next(size - data_length, name)
            batch_data = np.concatenate((batch_data, additional_data), axis=0)
            batch_labels = np.concatenate((batch_labels, additional_labels), axis=0)

        return batch_data, batch_labels

class TwitterApiDataProvider(object):
    def __init__(self,
            consumer_key,
            consumer_secret,
            access_token,
            access_secret):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.tweets_buckets = None
        self.file_name = "tweets.data.json"

    def read(self, search):
        search_words = search.split(" ")
        if not self.tweets_buckets: self._read_data()

        tweets_text = []
        for search_word in search_words:
            tweets_bucket = self.tweets_buckets.get(search_word, {})
            since_id = tweets_bucket.get("since_id", None)
            tweets = tweets_bucket.get("tweets", [])
            new_since_id, new_tweets = self._request_tweets(search_word, since_id=since_id)
            tweets += new_tweets
            tweets_bucket["since_id"] = new_since_id
            tweets_bucket["tweets"] = tweets
            tweets_text.append(map(lambda t: t["text"], tweets_bucket))
            self.tweets_buckets[search_word] = tweets_bucket

        self._write_data()
        return tweets_text

    def _request_tweets(self, search_word, since_id):
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)

        tweets = []
        api = tweepy.API(auth)
        max_id = None
        new_since_id = None
        total = 0
        logger.info("start search by %s" % search_word)
        while True:
            tweets_batch = api.search(search_word, max_id=max_id, since_id=since_id)
            logger.info("get " + len(tweets_batch) + " tweets by '" + search_word + "'")
            if not new_since_id: new_since_id = tweets_batch.since_id
            if max_id == tweets_batch.max_id: break

            max_id = tweets_batch.max_id
            total += len(tweets_batch)

            for tweet in tweets_batch:
                tweets.append(tweet._json)

            if not max_id:
                break

        logger.info("done with search found %s new tweets" % total)
        return new_since_id, tweets

    def _read_data(self):
        with open(self.file_name, 'r') as file:
            self.tweets_buckets = json.loads(file.read())

    def _write_data(self):
        with open(self.file_name, 'w') as file:
            file.write(json.dumps(self.tweets_buckets, indent=4))


class TwitterTrainingDataProvider(object):
    def __init__(self, amount=1):
        self.amount = amount
        self.file_name = "Sentiment140.tenPercent.sample.tweets.tsv"

    def read(self):
        logger.info(" read tweets from file raw file")
        with codecs.open(self.file_name, 'r', 'utf-8') as file:
            lines = file.readlines()
            neg_count = 0
            pos_count = 0
            neg_re = re.compile('\s+negative$')
            pos_re = re.compile('\s+positive$')
            texts = []
            labels = []
            toread = int(len(lines) * self.amount)
            lines = lines[:toread]

            for line in lines:
                if pos_re.search(line):
                    pos_count += 1
                    labels.append(1)
                    texts.append(pos_re.sub("", line))
                elif neg_re.search(line):
                    neg_count += 1
                    labels.append(0)
                    texts.append(neg_re.sub("", line))

        logger.info(" positive: " + str(pos_count) + ", negative: " + str(neg_count))
        logger.info(" total: " + str(pos_count + neg_count))
        return texts, labels

