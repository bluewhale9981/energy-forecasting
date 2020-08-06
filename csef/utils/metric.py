# -*- coding: utf-8 -*-
"""The metric (scikit learn) collection utils."""

from __future__ import absolute_import

from csef.utils.helper import load_class


METRIC_MAPPINGS = {
    # Keep the fullname
    'auc': 'sklearn.metrics.roc_auc_score',
    'rmse': 'sklearn.metrics.mean_squared_error'
}


def get_metric_func(metric_name):
    """
    Get the metric function based on the metric name.
    :param metric_name: The name of the metric (e.g. auc, rmse ...)
    :return: The metric function
    """
    return load_class(METRIC_MAPPINGS[metric_name])

