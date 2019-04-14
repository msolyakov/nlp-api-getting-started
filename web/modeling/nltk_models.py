from __future__ import absolute_import
# NLTK Wrapper Class 

import logging

log = logging.getLogger(__name__)

class NltkWrapper():

    is_model_learned = False

    def init_app(self, flask_app):
        log.info('>>>>> NLTK initialization started')
        # TODO
        self.ensure_model_is_learned()
        log.info('>>>>> NLTK initialization completed')

    def ensure_model_is_learned(self):
        if not self.is_model_learned:
            # TODO
            log.info('>>>>> NLTK Model is learned now')
            self.is_model_learned = True
        else:
            # Nothing TODO
            pass

    def do_sentiment(self, text):
        self.ensure_model_is_learned()
        # TODO
        score = 2
        return score


nltk = NltkWrapper()

