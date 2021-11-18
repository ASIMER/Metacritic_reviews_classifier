from time import time

from reviews_classifier.service import bert_model, bert_tokenizer, \
    classifier_model
from reviews_classifier.service.bert import vectorize
from reviews_classifier.service.text_filtering import preprocess
from reviews_classifier.service.translate import translate_review
import numpy as np
import logging
from flask import current_app as app


def generator(data_set,
              labels_set=None,
              batch_size=128,
              vectorizer=None,
              three_d_tensor_mode=False,
              predict_mode=False,
              simple_return=False,
              vec_len=4096,
              hidden_layers_lstm=4,
              vec_len_lstm=1024):
    """
    Generate data batches

    data_set      - data with features
                    Types: itterable[str] or itterable[itterable]
                    (ONLY itterable (feature vector)
                    if vectorizer is not defined)
    label_set     - data with classes (itterable)
    batch_size    - size of batches
    vectorizer    - custom vectorizer (callable),
                    with expected iterable return (list, array)
    predict_mode  - generate only features, without classes
    simple_return - return one x or sequence
    """
    if not isinstance(data_set, (list, np.ndarray)):
        # convert series to list
        data_set = data_set.to_list()
    if not predict_mode and not isinstance(data_set, (list, np.ndarray)):
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

        if not isinstance(data_set, np.ndarray):
            # save part of data
            x = np.asarray(data_set[initial:final]).astype(np.float32)
            if not predict_mode:
                y = np.asarray(labels_set[initial:final]).astype(np.float32)
        else:
            # save part of data
            x = data_set[initial:final]
            if not predict_mode:
                y = labels_set[initial:final].astype(np.float32)

        batch_number = (batch_number+1) % items_per_epoch
        x = x[:, :vec_len]
        # x = x + 1
        if three_d_tensor_mode:

            # x = np.expand_dims(x, -1)
            x = x.reshape((x.shape[0], hidden_layers_lstm, vec_len_lstm))
            # x = np.expand_dims(x, -1)

        if simple_return:
            if predict_mode:
                yield x
            else:
                yield x, y
        else:
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

    app.logger.info('Start text translating')
    start_time = time()
    try:
        text, language = translate_review(text)
    except:
        app.logger.error(f'Translate error with text: {text}')
        return None, 'Translate error'
    app.logger.info(f'Text translated, spend time: {time() - start_time}')

    app.logger.info('Start text preprocessing')
    start_time = time()
    try:
        text = preprocess(text)
    except:
        app.logger.error(f'Preprocess error with text: {text}')
        return None, 'Preprocess error'

    app.logger.info(f'Text preprocessed, spend time: {time() - start_time}')

    start_time = time()
    app.logger.info('Start vectorizing text with BERT')
    try:
        vector = vectorize(text,
                           bert_model=bert_model,
                           bert_tokenizer=bert_tokenizer)
        if isinstance(vector, type(None)):
            return None, 'BERT vectirizing error'
    except:
        app.logger.error(f'BERT vectirizing error with text: {text}')
        return None, 'BERT vectirizing error'
    app.logger.info(f'Text vectorized, spend time: {time() - start_time}')

    g = generator([vector], predict_mode=True)

    start_time = time()
    app.logger.info('Start predicting text scrore')
    try:
        score = classifier_model.predict(g, steps=1)
    except:
        app.logger.error(f'Classification error with text: {text}')
        return None, 'Classification error'
    # convert np.array to score
    score = round(float(score[0][0]), 3)
    # cut border values
    score = 10 if score > 10 else score
    score = 0 if score < 0 else score
    """try:
        score = classifier_model.predict(g, steps=1)
        # convert np.array to score
        score = round(float(score[0][0]), 3)
    except:
        print('Error')
    """
    app.logger.info(f'Text score predicted, spend time: {time() - start_time}')
    return score, language
