from __future__ import absolute_import

from flask import request
from flask_restplus import Resource
from web.api.restplus import api

from web.bll.core import do_sentiment_tb
from web.api.models.common import text_argument
from web.api.models.sentiment import sentiment_result

# import logging
# log = logging.getLogger()

ns = api.namespace('v1', description='TextBlob models')

@ns.route('/sentiment')
class PostsCollection(Resource):

    @api.expect(text_argument)
    @api.marshal_with(sentiment_result)
    def post(self):
        text = text_argument.parse_args(request).get('text')
        return do_sentiment_tb(text)