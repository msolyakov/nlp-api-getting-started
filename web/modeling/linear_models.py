from __future__ import absolute_import
# Linear Model Wrapper Class 

# import numpy as np
from sklearn.linear_model import LogisticRegression 
from web.data.en.datasets import SentimentLabelledWithTdIdfDataset

import logging

log = logging.getLogger()

class LinearModelWrapper():

    def __init__(self):
        self.log = logging.getLogger()
        self.is_model_trained = False
        self.classifier = None    

    def init_app(self):
        log.info('>>>>> LogisticRegression initialization started')
        self.ensure_model_is_trained()
        log.info('>>>>> LogisticRegression initialization completed')

    def ensure_model_is_trained(self):
        if not self.is_model_trained:

            ds = SentimentLabelledWithTdIdfDataset()
            ds.load_data()

            # train the classifier
            self.log.info('>>>>> Create LogisticRegression, train it')       

            self.classifier = LogisticRegression().fit(ds.train_x, ds.train_y)

            # test the accuracy
            acr = self.classifier.score(ds.test_x, ds.test_y)
            self.log.info(str.format('>>>>> LogisticRegression trained with accuracy {}', acr))           

            log.info('>>>>> LogisticRegression is learned now')
            self.is_model_trained = True

        return self.classifier


    def do_predict(self, x_text):
        clf = self.ensure_model_is_trained()
        predicted = clf.predict(x_text) 

        return predicted

