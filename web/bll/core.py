from __future__ import absolute_import
# Business layer

from web.bll.models import SentimentResult
from web.modeling.textblob_models import TextBlobWrapper

model = TextBlobWrapper()

def init_app():
    # Train model before start
    model.init_app()

def do_sentiment(text):
    rate = model.do_sentiment_classification(text) # Do sentiment analysis for specified text 
    return SentimentResult(text, rate)
