from __future__ import absolute_import

import unittest
import loggingtestcase
from web.data.en.datasets import SentimentLabelledWithTdIdfDataset

class DataSetTests(unittest.TestCase):

    @loggingtestcase.capturelogs(None, level='INFO')
    def test_train_classifier(self, logs):
        ds = SentimentLabelledWithTdIdfDataset()
        ds.load_data()

        print(ds.train[:0])
        print(ds.train[:1])
        
        self.assertIsNotNone(ds.train)


# from sklearn.datasets import fetch_20newsgroups

# newsgroups_train = fetch_20newsgroups(subset='train')

# print(newsgroups_train.data[:1])

# print(list(newsgroups_train.target_names))

# print(list(newsgroups_train.filenames[:20]))

# print(list(newsgroups_train.target[:20]))