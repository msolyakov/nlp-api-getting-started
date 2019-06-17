from __future__ import absolute_import
# Data Splitters

import nltk
from sklearn import feature_extraction
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

        return x_train, x_test, y_train, y_test 


class TdIdfDataSplitter():

    def __init__(self, text_attr, rate_attr, test_part_size=.3):
        self.text_attr = text_attr
        self.rate_attr = rate_attr
        self.test_part_size = test_part_size

    def extract_tdidf(self, corpus):
        # Extract TF-IDF features from corpus
        # vectorize means we turn non-numerical data into an array of numbers
        count_vectorizer = feature_extraction.text.CountVectorizer(
            lowercase = True,  # for demonstration, True by default
            tokenizer = nltk.word_tokenize,  # use the NLTK tokenizer
            stop_words = 'english',  # remove stop words
            min_df = 1  # minimum document frequency, i.e. the word must appear more than once.
        )
        processed_corpus = count_vectorizer.fit_transform(corpus)
        return feature_extraction.text.TfidfTransformer().fit_transform(processed_corpus)        
    
    def split_data(self, data):
        # create x and y and split them to validate the model
        x = data[self.text_attr]
        y = data[self.rate_attr]

        x_tfidf = self.extract_tdidf(x)

        log.info('>>>>> Spliting data to train and test sets.')
        x_train, x_test, y_train, y_test = train_test_split(x_tfidf, y, test_size=self.test_part_size, random_state=42)

        return x_train, x_test, y_train, y_test 
 