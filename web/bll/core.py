from __future__ import absolute_import
# Business layer

from web.bll.models import SentimentResult
from web.modeling.nltk_models import nltk

def do_sentiment(text):
    score = nltk.do_sentiment(text) 
    return SentimentResult(text, score)



