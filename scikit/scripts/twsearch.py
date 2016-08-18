import inspect
import tweepy
import re
import json
from tweepy import OAuthHandler


def log(text):
    print ("="*20)
    print (text)
    print ("="*20)


def to_json(status):
    tweet = json.dumps(status._json)
    log(tweet)
    return tweet


def get_tweets_by_user(words, since_id=None):
    consumer_key = 'mUnZ9yDNN2pRwzyqzrkwjQ'
    consumer_secret = 'Ow9pJWZNzmg4TX1zrLxfQFnvBFpBi8CydxeQ3Xu6uM'
    access_token = '238066404-NDqnqYLV7rNO8QKTRw0hWUxiHqKHa4LyZp5ViKUT'
    access_secret = 'OEjieKwOLdXwSpss1DmNzLyucBfre3oWuKK1JNdD5wwC9'

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    data = {}
    api = tweepy.API(auth)
    words = words.split(" ")
    max_id = None
    since_id = None
    total = 0
    for word in words:
        log("start search by %s" % word)
        while True:
            tweets = api.search(word, max_id=max_id)
            if not since_id: since_id = tweets.since_id
            log("since : " + str(tweets.since_id))
            log("prev_max_id : " + str(max_id))
            log("current_max_id : " + str(tweets.max_id))
            if max_id == tweets.max_id:
                break
            max_id = tweets.max_id
            total += len(tweets)
            log("get %s tweets" % len(tweets))
            log("total %s tweets" % total)

            for tweet in tweets:
                data.setdefault(tweet.author.name, [])
                data[tweet.author.name].append(tweet._json)

            if not max_id:
                break

        log("done with search by %s" % word)
    return since_id, data


def get_words(text):
    txt = re.compile(r'<[^>]+>').sub('', text)
    words = re.compile(r'[^A-Z^a-z]+').split(txt)
    return list(filter(None, map(str.lower, words)))


def get_words_count(text):
    words = get_words(text)

    result = {}
    for word in words:
        result.setdefault(word, 0)
        result[word] += 1

        return result


def write_to_file(file_name, word_list, word_count_by_author):
    log("write file")
    log("words:%s" % len(word_list))
    log("authors:%s" % len(word_count_by_author))

    with codecs.open(file_name, 'w', 'utf-8') as file:
        file.write('tweets')
        for word in word_list:
            file.write('\t%s' % word)
            file.write('\n')

            for author, word_count in word_count_by_author.items():
                # author = str(author.encode('ascii','ignore'))
                file.write(author)
                for word in word_list:
                    if word in word_count:
                        file.write('\t%d' % word_count[word]) else: file.write('\t0')
                        file.write('\n')


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
    dataset = []
    for tweet in tweets:
        words_count = get_words_count(tweet)
        dataset.append(words_count)
        for word_count in words_count:
            words_appearance.setdefault(word_count.word, 0)
            words_appearance[word_count.word] += 1

    word_list = get_word_list(words_appearance, len(tweets))
    write_to_file("words.txt", word_list, dataset)

def run():
    tweets = get()

    words_appearance = {}
    for tweet in tweet:
        word_count = map(lambda t: get_words_count(t), tweets)
        word_count.append(word_count)
        for word_count in word_count:
            words_appearance.setdefault(word_count.word, 0)
            words_appearance[word] += 1

    word_list = get_word_list(words_appearance, len(word_count_by_author))
    write_to_file("words.txt", word_list, word_count_by_author)

def read_data(file_name):
    with codecs.open(file_name, 'r') as file:
        data = json.loads(file.read())
        log(data)
        return data

def write_data(file_name, data):
    with codecs.open(file_name, 'w') as file:
        file.write(json.dumps(data, indent=4))


data = read_data("data.txt")
since_id, tweets = get_tweets_by_user("dreamhost")
write_data("data.txt", data)

