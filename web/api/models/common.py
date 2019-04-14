from __future__ import absolute_import
# Core data contracts and requests parsers definitions

from flask_restplus import fields, reqparse
from web.api.restplus import api

# Text Request parser  
text_argument = reqparse.RequestParser()
text_argument.add_argument('text', type=str, required=True, help='Text to analyse')


# Page Request parser
pagination_arguments = reqparse.RequestParser()
pagination_arguments.add_argument('page', type=int, required=False, default=1, help='Page number')
pagination_arguments.add_argument('bool', type=bool, required=False, default=1, help='Page number')
pagination_arguments.add_argument('per_page', type=int, required=False, choices=[2, 10, 20, 30, 40, 50],
                                  default=10, help='Results per page {error_msg}')

# Paged Result model
pagination = api.model('A page of results', {
    'page': fields.Integer(description='Number of this page of results'),
    'pages': fields.Integer(description='Total number of pages of results'),
    'per_page': fields.Integer(description='Number of items per page of results'),
    'total': fields.Integer(description='Total number of results'),
})