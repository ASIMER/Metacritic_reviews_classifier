from time import time

from reviews_classifier.service import bert_model, bert_tokenizer
from reviews_classifier.service.bert import vectorize
from reviews_classifier.service.text_filtering import preprocess
import numpy as np

from flask import current_app as app


def generator(data_set,
              labels_set=None,
              batch_size=128,
              vectorizer=None,
              three_d_tensor_mode=False,
              predict_mode=False):
    """
    Generate data batches

    data_set     - data with features
                   Types: itterable[str] or itterable[itterable]
                   (ONLY itterable (feature vector) if vectorizer is not defined)
    label_set    - data with classes (itterable)
    batch_size   - size of batches
    vectorizer   - custom vectorizer (callable),
                   with expected iterable return (list, array)
    predict_mode - generate only features, without classes
    """
    if not isinstance(data_set, list):
        # convert series to list
        data_set = data_set.to_list()
    if not predict_mode and not isinstance(data_set, list):
        labels_set = labels_set.to_list()

    # count batches per epoch, if generator will run more then one epoch
    batch_number = 0
    data_set_len = len(data_set)
    items_per_epoch = int(data_set_len/batch_size) or 1
    # vectorize data
    if vectorizer:
        data_set = data_set.map(vectorizer)

    while True:
        initial = (batch_number*batch_size) % data_set_len
        final = initial + batch_size

        # save part of data
        x = np.asarray(data_set[initial:final]).astype(np.float32)
        if not predict_mode:
            y = np.asarray(labels_set[initial:final]).astype(np.float32)

        batch_number = (batch_number+1) % items_per_epoch
        x = x[:, :1024]
        if three_d_tensor_mode:

            x = np.expand_dims(x, -1)
            #x = x.reshape((batch_size, 4, 1024))
        """
        if predict_mode:
            yield x
        else:
            yield x, y
        """
        if predict_mode:
            yield [x, x]
        else:
            yield [x, x], y


def predict(text) -> float:
    """
    generate predicted score

    :param text: text from user
    :return: predicted score
    """
    vector = 'Error'
    app.logger.info('Start text preprocessing')
    start_time = time()
    text = preprocess(text)
    app.logger.info(f'Text preprocessed, spend time: {time() - start_time}')

    start_time = time()
    app.logger.info('Start vectorizing text with BERT')
    try:
        vector = vectorize(text,
                           bert_model=bert_model,
                           bert_tokenizer=bert_tokenizer)
    except:
        vector = 'Error'
    app.logger.info(f'Text vectorized, spend time: {time() - start_time}')

    return vector
