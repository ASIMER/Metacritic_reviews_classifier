import os
import sys

from tensorflow import keras
from transformers import BertModel, BertTokenizer
import logging
from time import time
import torch
import google.cloud.logging # Don't conflict with standard logging
from google.cloud.logging.handlers import CloudLoggingHandler

client_logger = google.cloud.logging.Client()
cloud_handler = CloudLoggingHandler(client_logger)
is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")
# create logger
logger = logging.getLogger('gunicorn.error') \
    if is_gunicorn else logging.getLogger(__name__)
logger.setLevel(logger.level if is_gunicorn else logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(cloud_handler)

if not is_gunicorn:
    pass


def init_bert():
    """Initialize BERT model"""
    start_init_time = time()
    logger.info('Start bert initializing')
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    # Load pre-trained model (weights)
    # Whether the model returns all hidden-states.
    model = BertModel.from_pretrained('bert-large-uncased',
                                      output_hidden_states=True,
                                      )
    # move to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.warning('BERT STARTED ON '
                    + torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "BERT STARTED ON CPU")

    logger.info(f'Bert initialized, spend time: {time()-start_init_time}')
    return model, tokenizer, device


def dif_pred(y_true, y_pred):
    return abs(y_true - y_pred)
def acc_pred(y_true, y_pred, similarity=2):
    return int(abs(y_true - y_pred) <= similarity)

def init_classifier():
    """Initialize TensorFlow model"""

    start_init_time = time()
    logger.info('Start classifier model initializing')
    model = keras.models.load_model(
            os.environ.get('CLASSIFIER_MODEL'),
            custom_objects={
                    'dif_pred': dif_pred,
                    'acc_pred': acc_pred,
                    }
            )
    logger.info(f'Classifier model initialized, '
                f'spend time: {time()-start_init_time}')
    return model


# initialize bert model
bert_model, bert_tokenizer, device = init_bert()
# initialize classifier
classifier_model = init_classifier()
