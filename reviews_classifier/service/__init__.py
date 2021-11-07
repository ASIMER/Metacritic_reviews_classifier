import os

from tensorflow import keras
from transformers import BertModel, BertTokenizer
import logging
from time import time


def init_bert():
    """Initialize BERT model"""
    gunicorn_logger = logging.getLogger('gunicorn.error')
    start_init_time = time()
    gunicorn_logger.info('Start bert initializing')
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    # Load pre-trained model (weights)
    # Whether the model returns all hidden-states.
    model = BertModel.from_pretrained('bert-large-uncased',
                                      output_hidden_states=True,
                                      )

    gunicorn_logger.info(f'Bert initialized, spend time: {time()-start_init_time}')
    return model, tokenizer


def init_classifier():
    """Initialize TensorFlow model"""
    gunicorn_logger = logging.getLogger('gunicorn.error')
    start_init_time = time()
    gunicorn_logger.info('Start classifier model initializing')
    model = keras.models.load_model(os.environ.get('CLASSIFIER_MODEL'))
    gunicorn_logger.info(f'Classifier model initialized, '
                 f'spend time: {time()-start_init_time}')
    return model


# initialize bert model
bert_model, bert_tokenizer = init_bert()
classifier_model = init_classifier()
