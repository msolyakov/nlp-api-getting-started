from __future__ import absolute_import

import unittest
import loggingtestcase

import numpy as np # linear algebra
import pandas as pd # data processing
from web.modeling.textblob_models import TextBlobWrapper
from web.data.en.datasets import AmazonAlexaDataset

class TextBlobWrapperTests(unittest.TestCase):

    @loggingtestcase.capturelogs(None, level='INFO')
    def test_train_classifier(self, logs):
        tb = TextBlobWrapper()
        tb.init_app()

        print('\nModel training is done. Logs:')
        for r in logs.records:
            print(r.message)
        
        self.assertIsNotNone(tb.classifier)

    @loggingtestcase.capturelogs(None, level='INFO')
    def test_classifier_on_separate_set(self, logs):
        tb = TextBlobWrapper()
        ds = AmazonAlexaDataset()
        ds.load_data()

        # Check poisitives
        correct_answers = 0
        data = ds.data.to_numpy()

        seach_mask = np.isin(data[:, 1], ['pos'])
        data = data[seach_mask][:100]

        for e in data[:]:
            # Model train will be performed on first classification call
            r = tb.do_sentiment_classication(e[0])
            if r == e[1]:
                correct_answers += 1

        self.assertLessEqual(correct_answers, 100)
        print(str.format('\nPositive answers prediction - {} of 100', correct_answers))

        total_correct_answers = correct_answers

        # Check negatives
        correct_answers = 0
        data = ds.data.to_numpy()

        seach_mask = np.isin(data[:, 1], ['neg'])
        data = data[seach_mask][:100]

        for e in data[:]:
            # Model train will be performed on first classification call
            r = tb.do_sentiment_classication(e[0])
            if r == e[1]:
                correct_answers += 1

        self.assertLessEqual(correct_answers, 100)
        print(str.format('\nNegative answers prediction - {} of 100', correct_answers))
        
        total_correct_answers += correct_answers
        print(str.format('\nTotal prediction score - {}', round( total_correct_answers/200, 4 ) ))


if __name__ == '__main__':
    unittest.main()