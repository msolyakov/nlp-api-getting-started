from __future__ import absolute_import
# Dataset loaders (csv/tsv)

import os
import logging
from web.data.loaders import CsvSentimentDataLoader
from web.data.splitters import SimpleDataSplitter

log = logging.getLogger()

class AmazonAlexaDataset():
    train = None
    test = None

    def load_data(self):
        # load the dataset
        file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'amazon_alexa/train.tsv'))
        delim = '\t'
        text_attr = 'verified_reviews'
        rate_attr = 'feedback'
 
        loader = CsvSentimentDataLoader(file_path, delim, text_attr, rate_attr, [1])
        splitter = SimpleDataSplitter(text_attr, rate_attr, test_part_size=.3)

        self.train, self.test = splitter.split_data(loader.load_data())


class SentimentLabelledDataset():
    train = None
    test = None

    def load_data(self):
        # load the dataset
        file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'sentiment_labelled/train.tsv'))
        delim = '\t'
        text_attr = 'review'
        rate_attr = 'feedback'

        loader = CsvSentimentDataLoader(file_path, delim, text_attr, rate_attr, ['1'])
        splitter = SimpleDataSplitter(text_attr, rate_attr, test_part_size=.3)

        self.train, self.test = splitter.split_data(loader.load_data())


class Twitter100kDataset():

    train = None
    test = None

    def load_data(self):
        # load the dataset
        file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'twitter_100k/train.csv'))
        delim = ','
        text_attr = 'SentimentText'
        rate_attr = 'Sentiment'

        loader = CsvSentimentDataLoader(file_path, delim, text_attr, rate_attr, ['1'])
        splitter = SimpleDataSplitter(text_attr, rate_attr, test_part_size=.3)

        self.train, self.test = splitter.split_data(loader.load_data())
