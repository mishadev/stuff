import inspect
import tweepy
import re
from tweepy import OAuthHandler

def log(text):
	print ("="*20)
	print (text)
	print ("="*20)

def get_tweets_by_user(words):
	consumer_key = 'mUnZ9yDNN2pRwzyqzrkwjQ'
	consumer_secret = 'Ow9pJWZNzmg4TX1zrLxfQFnvBFpBi8CydxeQ3Xu6uM'
	access_token = '238066404-NDqnqYLV7rNO8QKTRw0hWUxiHqKHa4LyZp5ViKUT'
	access_secret = 'OEjieKwOLdXwSpss1DmNzLyucBfre3oWuKK1JNdD5wwC9'
 
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_secret)
 
	text = {}
	api = tweepy.API(auth)
	words = words.split(" ")
	max_id = None
	total = 0
	for word in words:
		log("start search by %s" % word)
		while True:
			tweets = api.search(word, max_id=max_id, show_name=True)
			max_id = tweets.max_id
			total += len(tweets)
			log("get %s tweets" % len(tweets))
			log("total %s tweets" % total)

			for tweet in tweets:
				text.setdefault(tweet.author.name, [])
				text[tweet.author.name].append(tweet.text)

			if not max_id:
				log("done with search by %s" % word)
				break

	return text

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

	with open(file_name,'w') as file:
		file.write('tweets')
		for word in word_list: 
			file.write('\t%s' % word)
		file.write('\n')

		for author, word_count in word_count_by_author.items():
			# author = str(author.encode('ascii','ignore'))
			file.write(author)
			for word in word_list:
				if word in word_count: 
					file.write('\t%d' % word_count[word])
				else:
					file.write('\t0')
			file.write('\n')

def get_word_list(words_appearance, authors_count):
	if authors_count <= 1:
		return list(words_appearance.keys())

	white_list = []
	black_list = []
	for word, app_count in words_appearance.items():
		probability = float(app_count) / authors_count
		one_app = 1 / authors_count
		if probability > one_app and probability < 0.7:
			white_list.append(word)
		else:
			log(word + " : " + str(probability))

	return white_list

def run():
	users = get_tweets_by_user("dreamhost")
	
	word_count_by_author = {}
	words_appearance = {}
	for author, tweets in users.items():
		word_count = get_words_count(" ".join(tweets))
		word_count_by_author[author] = word_count
		for word, count in word_count.items():
			words_appearance.setdefault(word, 0)
			words_appearance[word] += 1

	word_list = get_word_list(words_appearance, len(word_count_by_author))
	write_to_file("words.txt", word_list, word_count_by_author)

run()
