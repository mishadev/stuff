import codecs
import unittest
import numpy as np
from dataproviders import TwetterDataProvider, TextAnalizer, TokensFilter, BatchDataReader
import logging
import sys
from featureshasing import FeatureHasher
from tokenizers import TweetTokenizerPlus

test_tweets = [
"Looking for a domain+hosting?an  the Try searching on @DreamHost and see what they recommend https://t.co/VxIbWPARc9",
"RT @xyz: Looking for a domain+hosting? Try searching on @DreamHost and see what they recommend https://t.co/VxIbWPARc9",
"@vaLewee I know!  Saw it on the news!",
"RT @jainsudhir: #Dreamhost - DEDICATED SERVER WEB HOSTING starting at $99/month. https://t.co/LBVWuGtOw2 -  Review - https://t.co/J0m5cxq6xD",
"@aubreyoday... I love that show...weeds! But i dont get it here in london  i miss my American tv shows....",
"RT @DreamHost: \"Earning Income from Your Blog Does Not Have to be a Daydream\" - https://t.co/oZvp7SwbYO via @blpro #blogging https://t.co/P\u2026",
"#3hotwords &quot;I love life&quot; WILLl  &amp; &quot;follow meeeeee pleeeeeeease&quot;",
"@chadetennant @DreamHost \u2026really helped me steer clear! Would you happen to know if there's a way I can donate to your competitor(s)?",
"@chadetennant Thank you, obvious spammer! I hate patronizing businesses with shady tactics like @DreamHost; you've\u2026 https://t.co/VogD0M3G53"
]

class TwetterDataProviderTest(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger("SomeTest.testSomething")

    def test_tweets_data(self):
        # analize form raw data file and save results to cache file
        data_provider = TwetterDataProvider(amount=0.001)
        reader = BatchDataReader()
        reader.use_data(*data_provider.fetch_data())
        labels, data = reader.next(2)
        rows_count = len(data)
        columns_count = len(data[0])
        self.assertEqual((rows_count, columns_count), (2, 98))

    def test_filter_entropy(self):
        f = TokensFilter()
        tokens_data = [
                ['I', 'hate', 'you'],
                ['I', 'love', 'life'],
                ['I', 'hurt', 'badly'],
                ['I', 'love', 'you'],
                ['I', 'hate', 'code'],
                ['I', 'stupid'],
                ['you', 'are', 'stupid'],
                ['I', 'badly', 'damage'],
                ['code', 'love', 'damage']
            ]
        labels = [
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [1, 0]
            ]
        actual = list(f.filter_tokens(tokens_data, labels))
        expected = ['badly', 'love', 'stupid', 'hate']
        self.assertEqual(actual, expected)

    def test_tokenizer_remove_usernames(self):
        tokenizer = TweetTokenizerPlus()
        actual = tokenizer._remove_handles(test_tweets[-1])
        expected = " Thank you, obvious spammer! I hate patronizing businesses with shady tactics like ; you've\\u2026 https://t.co/VogD0M3G53"
        self.assertEqual(actual, expected)

    def test_tokenizer_remove_links(self):
        tokenizer = TweetTokenizerPlus()
        actual = tokenizer._remove_links(test_tweets[0])
        expected = "Looking for a domain+hosting?an  the Try searching on @DreamHost and see what they recommend "
        self.assertEqual(actual, expected)

    def test_tokenizer2(self):
        tokenizer = TweetTokenizerPlus()
        actual = tokenizer.tokenize(test_tweets[2])
        expected = [u'I', u'know', u'Saw', u'news']
        self.assertEqual(actual, expected)

    def test_tokenizer3(self):
        tokenizer = TweetTokenizerPlus()
        actual = tokenizer.tokenize(test_tweets[3])
        expected = [u'Dreamhost', u'DEDICATED', u'SERVER', u'WEB', u'HOSTING', u'starting', '99', u'month', u'Review']
        self.assertEqual(actual, expected)

    def test_tokenizer0(self):
        tokenizer = TweetTokenizerPlus()
        actual = tokenizer.tokenize(test_tweets[0])
        expected = [u'Looking', u'domain', u'hosting', u'Try', u'searching', u'see', u'recommend']
        self.assertEqual(actual, expected)

    def test_analize(self):
        analizer = TextAnalizer(use_cache=False, use_pos_tag=True)
        texts = [
                'I hate you',
                'I love life',
                'I love you',
                'I hate life'
            ]
        labels = [
                [0, 1],
                [1, 0],
                [1, 0],
                [0, 1]
            ]
        expected_columns = [u'love/VBP', u'I/PRP love/VBP', u'I/PRP hate/VBP', u'hate/VBP']
        expected_row = [
                [[0], [0], [1], [1]],
                [[1], [1], [0], [0]],
                [[1], [1], [0], [0]],
                [[0], [0], [1], [1]]
            ]
        columns, data = analizer.analize(texts, labels)
        self.assertEqual(columns, expected_columns)
        self.assertTrue(np.array_equal(expected_row, data))

    def test_inner_analize(self):
        analizer = TextAnalizer(use_pos_tag=True)
        measures = analizer._analize([test_tweets[-3]])
        self.assertEqual(measures, ([
            [
                u'3hotwords/NNS',
                u'I/PRP',
                u'love/VBP',
                u'life/NN',
                u'WILLl/NNP',
                u'follow/VBP',
                u'mee/NN',
                u'pleease/NN',
                u'3hotwords/NNS I/PRP',
                u'I/PRP love/VBP',
                u'love/VBP life/NN',
                u'life/NN WILLl/NNP',
                u'WILLl/NNP follow/VBP',
                u'follow/VBP mee/NN',
                u'mee/NN pleease/NN'
            ]
        ]))

    def test_create_no_pos_tagging(self):
        hasher = FeatureHasher(use_pos_tag=False)
        actual = hasher.create(["How", "bad", "is", "it", "or", "not", "so", "bad"])
        expected = [
            'How', 'bad', 'is', 'it', 'or', 'not', 'so', 'bad',
            'How bad', 'bad is', 'is it', 'it or not', 'or not so', 'not so bad'
        ]
        self.assertEqual(actual, expected)

    def test_create(self):
        hasher = FeatureHasher()
        actual = hasher.create(["How", "bad", "is", "it", "or", "not", "so", "bad"])
        expected = [
            'How/WRB',
            'bad/JJ',
            'is/VBZ',
            'it/PRP',
            'or/CC',
            'not/RB',
            'so/RB',
            'bad/JJ',
            'How/WRB bad/JJ',
            'bad/JJ is/VBZ',
            'is/VBZ it/PRP',
            'it/PRP or/CC not/RB',
            'or/CC not/RB so/RB',
            'not/RB so/RB bad/JJ'
            ]
        self.assertEqual(actual, expected)

    def test_lexicon_features(self):
        hasher = FeatureHasher()
        _, actual_pos, actual_neg =(
            hasher.lexicon_features(["bad", "worse", "worst", "good", "best"]))

        expected = [2, 3]
        self.assertEqual([actual_pos, actual_neg], expected)

    def test_tags_not(self):
        hasher = FeatureHasher()
        actual = hasher.pos_tagging(["How", "don't", "you", "do"])
        expected = ["How/WRB", "don't/NN", "you/PRP", "do/VBP"]
        self.assertEqual(actual, expected)

    def test_tags(self):
        hasher = FeatureHasher()
        actual = hasher.pos_tagging(["How", "do", "you", "do"])
        expected = ['How/WRB', 'do/VB', 'you/PRP', 'do/VB']
        self.assertEqual(actual, expected)

    def test_nt_attach(self):
        hasher = FeatureHasher()
        actual = hasher.bi_gramm(['How/WRB', 'can\'t/VB', 'you/PRP', 'do/VB'])
        expected = ['How/WRB can\'t/VB not', 'can\'t/VB not you/PRP', 'not you/PRP do/VB']
        self.assertEqual(actual, expected)

    def test_nt_in_the_last_and_first(self):
        hasher = FeatureHasher()
        actual = hasher.bi_gramm(['don\'t/RB'])
        expected = ['don\'t/RB not']
        self.assertEqual(actual, expected)

    def test_not_in_the_last_and_first(self):
        hasher = FeatureHasher()
        actual = hasher.bi_gramm(['not/RB'])
        expected = ['not/RB']
        self.assertEqual(actual, expected)

    def test_not_in_the_last(self):
        hasher = FeatureHasher()
        actual = hasher.bi_gramm(['How/WRB', 'not/RB'])
        expected = ['How/WRB not/RB']
        self.assertEqual(actual, expected)

    def test_not_in_the_begin_attach(self):
        hasher = FeatureHasher()
        actual = hasher.bi_gramm(['How/WRB', 'not/RB', 'you/PRP', 'do/VB'])
        expected = ['How/WRB not/RB you/PRP', 'not/RB you/PRP do/VB']
        self.assertEqual(actual, expected)

    def test_not_in_the_start_attach(self):
        hasher = FeatureHasher()
        actual = hasher.bi_gramm(['not/RB', 'How/WRB', 'you/PRP', 'do/VB'])
        expected = ['not/RB How/WRB you/PRP', 'you/PRP do/VB']
        self.assertEqual(actual, expected)

    def test_not_in_the_middle_attach(self):
        hasher = FeatureHasher()
        actual = hasher.bi_gramm(['How/WRB', 'do/VB', 'not/RB', 'you/PRP', 'do/VB'])
        expected = ['How/WRB do/VB not/RB', 'do/VB not/RB you/PRP', 'not/RB you/PRP do/VB']
        self.assertEqual(actual, expected)

    def test_basic_twogramms(self):
        hasher = FeatureHasher()
        actual = hasher.bi_gramm(['How/WRB', 'do/VB', 'you/PRP', 'do/VB'])
        expected = ["How/WRB do/VB", "do/VB you/PRP", "you/PRP do/VB"]

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    unittest.main()

