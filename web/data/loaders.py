from __future__ import absolute_import
# Data Loaders

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # to split the training and testing data

import logging

log = logging.getLogger()

class CsvSentimentDataLoader():

    def __init__(self, file_path, delim, text_attr, rate_attr, pos_rates):
        self.data_path = file_path
        self.delimiter = delim
        self.text_attr = text_attr
        self.rate_attr = rate_attr
        self.pos_rates = pos_rates


    def load_train_test(self, test_part_size=.3):
    
        # load the dataset
        log.info(str.format('>>>>> Loading CSV/TSV data from {}', self.data_path))

        data = pd.read_csv(self.data_path, self.delimiter)
        data.head()

        data = data[[self.text_attr, self.rate_attr]]
        log.info(str.format('>>>>> Rate value type: {}', type(data[self.rate_attr][0])))

        log.info(str.format('>>>>> CSV/TSV data was loaded {}. Convert specifed rates to ''pos'' and ''neg''.', data.shape))
        data[self.rate_attr] = np.where(data[self.rate_attr].isin(self.pos_rates), 'pos', 'neg')

        log.info(str.format('>>>>> First five rows: \n{}', data[:5]))

        # create x and y and split them to validate the model
        x = data[self.text_attr]
        y = data[self.rate_attr]

        log.info('>>>>> Spliting data to train and test sets.')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_part_size)

        train = [x for x in zip(x_train, y_train)]
        test = [x for x in zip(x_test, y_test)]

        return train, test
