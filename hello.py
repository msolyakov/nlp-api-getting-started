from web.data.en.datasets import SentimentLabelledDataset, AmazonAlexaDataset
import logging
import logging.config

logging.config.fileConfig('logging.conf')
log = logging.getLogger()

SentimentLabelledDataset().load_data()
AmazonAlexaDataset().load_data()