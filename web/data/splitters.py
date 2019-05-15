from __future__ import absolute_import
# Data Splitters

from sklearn.model_selection import train_test_split # to split the training and testing data

import logging

log = logging.getLogger()

class SimpleDataSplitter():

    def __init__(self, text_attr, rate_attr, test_part_size=.3):
        self.text_attr = text_attr
        self.rate_attr = rate_attr
        self.test_part_size = test_part_size
    
    def split_data(self, data):
        # create x and y and split them to validate the model
        x = data[self.text_attr]
        y = data[self.rate_attr]

        log.info('>>>>> Spliting data to train and test sets.')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_part_size)

        train = [x for x in zip(x_train, y_train)]
        test = [x for x in zip(x_test, y_test)]

        return train, test 