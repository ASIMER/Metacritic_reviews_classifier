import os

from tensorflow import keras
import logging
from time import time




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
classifier_model = init_classifier()
