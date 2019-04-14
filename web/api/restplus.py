from __future__ import absolute_import
# API bootstrap file

import logging
from web.api import settings
from flask_restplus import Api
# import traceback
# from sqlalchemy.orm.exc import NoResultFound

log = logging.getLogger()

api = Api(version='1.0', title='NLP API - Getting Started',
          description='A simple NLP API demonstration')

@api.errorhandler
def default_error_handler(e):
    message = 'An unhandled exception occurred.'
    log.exception(message)

    if not settings.FLASK_DEBUG:
        return {'message': message}, 500

# @api.errorhandler(NoResultFound)
# def database_not_found_error_handler(e):
#     log.warning(traceback.format_exc())
#     return {'message': 'A database result was required but none was found.'}, 404
