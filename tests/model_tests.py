from __future__ import absolute_import

import unittest
import loggingtestcase

import numpy as np # linear algebra
from web.modeling.textblob_models import TextBlobWrapper
from web.modeling.linear_models import LinearModelWrapper
from web.data.en.datasets import AmazonAlexaDataset

class LinearModelWrapperTests(unittest.TestCase):

    @loggingtestcase.capturelogs(None, level='INFO')
    def test_train_linear(self, logs):
        lin_mod = LinearModelWrapper()
        lin_mod.init_app()

        print('\nModel training is done. Logs:')
        for r in logs.records:
            print(r.message)
        
        self.assertIsNotNone(lin_mod.classifier)


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
        tb = TextBlobWrapper() # Going to be trained on Sentiment Labelled dataset 
        ds = AmazonAlexaDataset() # Test dataset
        ds.load_data()

        # Check poisitives
        true_pos = 0
        data = ds.data.to_numpy()

        seach_mask = np.isin(data[:, 1], ['pos'])
        data = data[seach_mask][:100]

        for e in data[:]:
            # Model train will be performed on first classification call
            r = tb.do_sentiment_classification(e[0])
            if r == e[1]:
                true_pos += 1

        self.assertLessEqual(true_pos, 100)
        print(str.format('\n\nTrue Positive answers - {} of 100', true_pos))

        # Check negatives
        true_neg = 0
        data = ds.data.to_numpy() # ds.data - DataFrame loaded from TSV/CVS

        seach_mask = np.isin(data[:, 1], ['neg'])
        data = data[seach_mask][:100]

        for e in data[:]:
            # Model train will be performed on first classification call
            r = tb.do_sentiment_classification(e[0])
            if r == e[1]:
                true_neg += 1

        self.assertLessEqual(true_neg, 100)
        print(str.format('\nTrue Negative answers - {} of 100', true_neg))
        
        print(str.format('\nPrediction accuracy - {}', round( ( true_pos + true_neg ) /200, 4 ) ))

        print('\nModel training logs:\n')
        for r in logs.records:
            print(r.message)    


if __name__ == '__main__':
    unittest.main()