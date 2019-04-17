from __future__ import absolute_import
# Data contracts definitions

from flask_restplus import fields
from web.api.restplus import api
from web.api.models.common import pagination

sentiment_result = api.model('Sentiment analysis result', {
    'text': fields.String(description='Text to analyse'),
    'rate': fields.String(description='Sentiment rate for specified text'),
})

sentiment_result_page = api.inherit('Page of sentiment analysis results', pagination, {
    'items': fields.List(fields.Nested(sentiment_result))
})