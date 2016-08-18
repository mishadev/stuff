import numpy as np
import codecs
import sys
import logging
from nns import RecurrentNN
from utils import Timer
from dataproviders import TwitterApiDataProvider, TwitterTrainingDataProvider, DatasetBuilder
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)

timer = Timer()
timer.checkin("start_getting_training_data")
dataset_builder = DatasetBuilder(
        use_pos_tag=True,
        use_single_words=True,
        use_lexicon_features=True)

data_provider = TwitterTrainingDataProvider(amount=1.0)
texts, labels = data_provider.read()
features_list, features, labels_list, labels = dataset_builder.build_labeled_dataset(texts, labels)
timer.checkin("end_getting_training_data")

config = {
        "accuracy_threshold": 0.94,
        "n_input": len(features_list),
        "n_steps": 1,
        "n_layers": 1,
        "n_hidden": 150,
        "n_classes": len(labels_list)
    }
nn = RecurrentNN(config)
nn.learn(features, labels)
timer.checkin("end_training")
logger.info("-"*50)
logger.info("ML :" + timer.diff("end_getting_training_data", "end_training"))
logger.info("Total :" + timer.diff("start_getting_training_data", "end_training"))

timer.checkin("end_training")
logger.info("-"*50)
logger.info("do some testing predictions")
api_tweet_provider = TwitterApiDataProvider(
        consumer_key='mUnZ9yDNN2pRwzyqzrkwjQ',
        consumer_secret='Ow9pJWZNzmg4TX1zrLxfQFnvBFpBi8CydxeQ3Xu6uM',
        access_token='238066404-NDqnqYLV7rNO8QKTRw0hWUxiHqKHa4LyZp5ViKUT',
        access_secret='OEjieKwOLdXwSpss1DmNzLyucBfre3oWuKK1JNdD5wwC9')

tweets = api_tweet_provider.read("dreamhost")
features_list, features_matrix, unclassifiable, labels_list = (
        dataset_builder.build_dataset(features_list[:-2], tweets))
if len(features_matrix):
    prediction = nn.predict(features_matrix)

with codecs.open("./predictions/prediction.csv", "w", "utf-8") as file:
    file.write('tweets')
    for label in labels_list:
        file.write('\t%s' % label)
    file.write('\n')

    for idx in range(0, len(prediction)):
        file.write(tweets[idx].encode('ascii', 'ignore'))
        for value in prediction[idx]:
            file.write('\t{:.6f}'.format(value))
        file.write('\n')

