# -*- coding: utf-8 -*-
from __future__ import absolute_import

from csef.data import preprocessing

"""The performance collection utils."""

import time
from math import sqrt
import multiprocessing
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


num_partitions = 5
num_cores = multiprocessing.cpu_count()


def parallelize_dataframe(df, func):
    data_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return df


class Timer(object):
    """
    Define a Time Class to computer total execution time.
    """
    def __init__(self):
        self.start = time.time()

    def restart(self):
        """
        Reset start time.
        """
        self.start = time.time()

    def get_time(self):
        """
        Get the total execution time.
        """
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = '%02d:%02d:%02d' % (h, m, s)

        return time_str


def measure_mae(actual, predicted):
    return mean_absolute_error(actual, predicted)


# walk-forward validation for univariate data
def walk_forward_validation(model_class, train, test, cfg, group_col='series_id', train_col='consumption'):
    errors = []

    # fit model
    model = model_class(cfg)
    model.fit(train)

    # do reset state before evaluation
    model.reset_states()

    for ser_id, ser_data in test.groupby(group_col):
        ser_train_data = train[train[group_col] == ser_id][train_col]

        _, _, scaler = preprocessing.prepare_training_data(ser_train_data, 24)
        yhat = model.predict(ser_train_data, scaler)

        ser_data_vals = ser_data[train_col].values
        error = measure_mae(ser_data_vals, yhat)

        print('Id: {}, Error: {}'.format(ser_id, error))

        errors.append(error)
        model.reset_states()

    # estimate prediction error
    error = np.mean(errors)
    print(' > %.3f' % error)
    return error, model


# repeat evaluation of a config
def repeat_evaluate(model_class, train, test, config, n_repeats=30):
    # fit and evaluate the model n times
    return [walk_forward_validation(model_class, train, test, config) for _ in range(n_repeats)]


# summarize model performance
def summarize_scores(name, scores):
    # print a summary
    scores_m, score_std = np.mean(scores), np.std(scores)
    print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))  # box and whisker plot
    plt.boxplot(scores)
    plt.show()
