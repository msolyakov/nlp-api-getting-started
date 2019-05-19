from __future__ import absolute_import
# Dataset loaders (csv/tsv)

import os
import logging
from web.data.loaders import CsvSentimentDataLoader
from web.data.splitters import SimpleDataSplitter

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
        self.train, self.test = splitter.split_data(self.data)


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
        self.train, self.test = splitter.split_data(self.data)


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
