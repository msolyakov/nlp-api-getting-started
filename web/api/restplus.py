from __future__ import absolute_import
# API bootstrap file

import logging
import traceback
import settings

from flask_restplus import Api
# from sqlalchemy.orm.exc import NoResultFound

log = logging.getLogger(__name__)

api = Api(version='1.0', title='NLP API - Getting Started',
          description='A simple NLP API demonstration')

@api.errorhandler
def default_error_handler(e):
    message = 'An unhandled exception occurred.'
    log.exception(message)

    if not settings.FLASK_DEBUG:
        return {'message': message}, 500

@api.errorhandler(404)
def not_found(e):
    log.warning(traceback.format_exc())
    return {'error': 'Not found', 'exception': e}, 404

# @api.errorhandler(NoResultFound)
# def database_not_found_error_handler(e):
#     log.warning(traceback.format_exc())
#     return {'message': 'A database result was required but none was found.'}, 404
