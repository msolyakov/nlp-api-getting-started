from __future__ import absolute_import

import logging

from flask import request
from flask_restplus import Resource
from web.api.restplus import api

from web.bll.core import do_sentiment
from web.api.models.common import text_argument
from web.api.models.sentiment import sentiment_result

log = logging.getLogger(__name__)

ns = api.namespace('api/sentiment', description='Sentiment analysis API operations')


@ns.route('/')
class PostsCollection(Resource):

    @api.expect(text_argument)
    @api.marshal_with(sentiment_result)
    def post(self):
        text = text_argument.parse_args(request).get('text')
        return do_sentiment(text)