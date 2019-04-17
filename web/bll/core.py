from __future__ import absolute_import
# Business layer

from web.bll.models import SentimentResult
from web.modeling.nltk_models import nltk
from web.modeling.textblob_models import tb

def do_sentiment_nltk(text):
    rate = nltk.do_sentiment(text) # Do sentiment analysis for specified text 
    return SentimentResult(text, rate)

def do_sentiment_tb(text):
    rate = tb.do_sentiment_classication(text) # Do sentiment analysis with Naive Bayes Classifier 
    return SentimentResult(text, rate)
