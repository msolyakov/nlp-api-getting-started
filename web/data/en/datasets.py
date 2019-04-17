from __future__ import absolute_import
# Dataset loaders (csv/tsv)

import os
import logging
from web.data.loaders import CsvSentimentDataLoader

log = logging.getLogger()

class AmazonAlexaDataset():
    train = None
    test = None

    def load_data(self):
        # load the dataset
        file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'amazon_alexa/train.tsv'))
        delim = '\t'

        loader = CsvSentimentDataLoader(file_path, delim, 'verified_reviews', 'feedback', [1])
        self.train, self.test = loader.load_train_test(test_part_size=.3)


class SentimentLabelledDataset():
    train = None
    test = None

    def load_data(self):
        # load the dataset
        file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'sentiment_labelled/train.tsv'))
        delim = '\t'

        loader = CsvSentimentDataLoader(file_path, delim, 'review', 'feedback', ['1'])
        self.train, self.test = loader.load_train_test(test_part_size=.3)


class Twitter100kDataset():

    train = None
    test = None

    def load_data(self):
        # load the dataset
        file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'twitter_100k/train.csv'))
        delim = ','

        loader = CsvSentimentDataLoader(file_path, delim, 'SentimentText', 'Sentiment', ['1'])
        self.train, self.test = loader.load_train_test(test_part_size=.3)
