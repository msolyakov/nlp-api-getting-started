from __future__ import absolute_import
# Dataset loaders (csv/tsv)

import os
import collections
import logging
from web.data.loaders import CsvSentimentDataLoader
from web.data.splitters import SimpleDataSplitter, TdIdfDataSplitter

log = logging.getLogger()

class AmazonAlexaDataset():

    def __init__(self):
        self.file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'amazon_alexa/train.tsv'))
        self.delim = '\t'
        self.text_attr = 'verified_reviews'
        self.rate_attr = 'feedback'
        self.pos_rates = [1]

        self.data = None
        self.train = None
        self.test = None

    def load_data(self):
        # load the dataset
        loader = CsvSentimentDataLoader(self.file_path, self.delim, self.text_attr, self.rate_attr, self.pos_rates)
        splitter = SimpleDataSplitter(self.text_attr, self.rate_attr, test_part_size=.3)

        self.data = loader.load_data()
        x_train, x_test, y_train, y_test = splitter.split_data(self.data)

        self.train = [x for x in zip(x_train, y_train)]
        self.test = [x for x in zip(x_test, y_test)]        


class SentimentLabelledDataset():

    def __init__(self):
        self.file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'sentiment_labelled/train.tsv'))
        self.delim = '\t'
        self.text_attr = 'review'
        self.rate_attr = 'feedback'
        self.pos_rates = ['1']

        self.data = None
        self.train = None
        self.test = None    

    def load_data(self):
        # load the dataset
        loader = CsvSentimentDataLoader(self.file_path, self.delim, self.text_attr, self.rate_attr, self.pos_rates)
        splitter = SimpleDataSplitter(self.text_attr, self.rate_attr, test_part_size=.3)

        self.data = loader.load_data()
        x_train, x_test, y_train, y_test = splitter.split_data(self.data)

        self.train = [x for x in zip(x_train, y_train)]
        self.test = [x for x in zip(x_test, y_test)]


class SentimentLabelledWithTdIdfDataset():

    def __init__(self):
        self.file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'sentiment_labelled/train.tsv'))
        self.delim = '\t'
        self.text_attr = 'review'
        self.rate_attr = 'feedback'
        self.pos_rates = ['1']

        self.data = None
        self.train_x = None
        self.test_x = None    
        self.train_y = None
        self.test_y = None    

    def load_data(self):
        # load the dataset
        loader = CsvSentimentDataLoader(self.file_path, self.delim, self.text_attr, self.rate_attr, self.pos_rates)
        splitter = TdIdfDataSplitter(self.text_attr, self.rate_attr, test_part_size=.3)

        self.data = loader.load_data()
        self.train_x, self.test_x, self.train_y, self.test_y = splitter.split_data(self.data)        


class Twitter100kDataset():
    
    def __init__(self):
        self.file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'twitter_100k/train.csv'))
        self.delim = ','
        self.text_attr = 'SentimentText'
        self.rate_attr = 'Sentiment'
        self.pos_rates = ['1']

        self.data = None
        self.train = None
        self.test = None    

    def load_data(self):
        # load the dataset
        loader = CsvSentimentDataLoader(self.file_path, self.delim, self.text_attr, self.rate_attr, self.pos_rates)
        splitter = SimpleDataSplitter(self.text_attr, self.rate_attr, test_part_size=.3)

        self.data = loader.load_data()
        self.train, self.test = splitter.split_data(self.data)
