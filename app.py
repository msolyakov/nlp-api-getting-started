from __future__ import absolute_import
#  Flask Application bootstrap file

import logging.config

from flask import Flask, Blueprint
from web.api import settings
from web.api.endpoints.v1 import ns as v1_api_namespace
from web.api.restplus import api
from web.modeling.nltk_models import nltk
from web.modeling.textblob_models import tb
# from rest_api_demo.api.nlp import ns as blog_categories_namespace
# from rest_api_demo.database import db

app = Flask(__name__)

logging.config.fileConfig('logging.conf')
log = logging.getLogger()


def configure_app(flask_app):
    flask_app.config['SERVER_NAME'] = settings.FLASK_SERVER_NAME
    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    flask_app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
    flask_app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
    flask_app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP
    #flask_app.config['SQLALCHEMY_DATABASE_URI'] = settings.SQLALCHEMY_DATABASE_URI
    #flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = settings.SQLALCHEMY_TRACK_MODIFICATIONS


def initialize_app(flask_app):
    configure_app(flask_app)

    blueprint = Blueprint('api', __name__, url_prefix='/api')
    api.init_app(blueprint)
    api.add_namespace(v1_api_namespace)
    flask_app.register_blueprint(blueprint)

    # Train model before start
    tb.init_app()
    # nltk.init_app()
    # db.init_app(flask_app)


def main():
    initialize_app(app)
    log.info(str.format('>>>>> Starting development server at http://{}/api/ <<<<<', app.config['SERVER_NAME']))
    app.run(debug=settings.FLASK_DEBUG)


if __name__ == "__main__":
    main()