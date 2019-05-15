from __future__ import absolute_import
# TextBlob Wrapper - https://textblob.readthedocs.io/en/dev/classifiers.html#classifying-text

import logging
from textblob.classifiers import NaiveBayesClassifier # classifier used to create model
from web.data.en.datasets import AmazonAlexaDataset


class TextBlobWrapper():

    log = None
    is_model_trained = False
    classifier = None


    def init_app(self):
        self.log = logging.getLogger()

        self.log.info('>>>>> TextBlob initialization started')
        self.ensure_model_is_trained()
        self.log.info('>>>>> TextBlob initialization completed')


    def ensure_model_is_trained(self):
        if not self.is_model_trained:

            ds = AmazonAlexaDataset()
            ds.load_data()

            # train the classifier
            self.log.info('>>>>> Create NaiveBayesClassifier, do feature extraction')              
            self.classifier = NaiveBayesClassifier(ds.train)

            # test the accuracy
            self.log.info('>>>>> Train NaiveBayesClassifier')              
            acr = self.classifier.accuracy(ds.test)
            self.log.info(str.format('>>>>> NaiveBayesClassifier trained with accuracy {}', acr))           

            self.is_model_trained = True

        return self.classifier


    def do_sentiment_classication(self, text):

        clf = self.ensure_model_is_trained()
        res = clf.classify(text)

        return res


tb = TextBlobWrapper()
