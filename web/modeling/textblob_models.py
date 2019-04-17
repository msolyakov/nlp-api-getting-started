from __future__ import absolute_import
# TextBlob Wrapper - https://textblob.readthedocs.io/en/dev/classifiers.html#classifying-text

import logging
from textblob.classifiers import NaiveBayesClassifier # classifier used to create model
from web.data.en.datasets import SentimentLabelledDataset


class TextBlobWrapper():

    log = None
    is_model_learned = False
    classifier = None    
    
    def init_app(self):
        self.log = logging.getLogger()

        self.log.info('>>>>> TextBlob initialization started')
        self.ensure_model_is_learned()
        self.log.info('>>>>> TextBlob initialization completed')


    def ensure_model_is_learned(self):
        if not self.is_model_learned:

            ds = SentimentLabelledDataset()
            ds.load_data()

            # train the classifier
            self.log.info('>>>>> Train NaiveBayesClassifier')              
            self.classifier = NaiveBayesClassifier(ds.train)

            # test the accuracy
            acr = self.classifier.accuracy(ds.test)
            self.log.info(str.format('>>>>> NaiveBayesClassifier trained with accurancy {}', acr))           

            self.is_model_learned = True

        return self.classifier


    def do_sentiment_classication(self, text):

        clf = self.ensure_model_is_learned()
        res = clf.classify(text)

        return res


tb = TextBlobWrapper()
