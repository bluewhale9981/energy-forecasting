# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from csef.transformer import DistanceBasedTransformer
from csef.utils.logging import getLogger
from csef.pipeline.base import BaseBlockPip
from sklearn.decomposition import PCA

logger = getLogger(logger_name=__name__)


class SelectKBestBlockPip(BaseBlockPip):
    """ This block used for select k best feature """

    X = None
    y = None
    data_test = None
    submission_ids = None

    config = {
        'k': 600
    }

    def _execute(self, inputs):

        # Inputs must have required fields
        assert 'X' in inputs, 'Input must have X'
        assert 'y' in inputs, 'Input must have y'

        # Local variables
        X = inputs['X']
        y = inputs['y']
        data_test = inputs.get('data_test')

        # Use selectK to select k number of best features
        selectK = SelectKBest(f_classif, k=int(self.config['k']))
        selectK.fit(X, y)

        # Filter the features
        feature_names = X.columns

        mask = selectK.get_support()  # list of booleans
        new_features = []  # The list of your K best features

        for is_bool, feature in zip(mask, feature_names):
            if is_bool:
                new_features.append(feature)

        # Cache the result for other can retrieve
        self.X = X[new_features]
        self.y = y
        self.submission_ids = inputs.get('submission_ids')
        self.data_test = data_test[new_features]

    def get_output(self):
        return {
            'X': self.X,
            'y': self.y,
            'data_test': self.data_test,
            'submission_ids': self.submission_ids
        }


class DistanceBasedBlockPip(BaseBlockPip):
    """ This block used for generating distance based features """

    X = None
    y = None
    data_test = None
    submission_ids = None

    config = {}

    def _execute(self, inputs):

        # Inputs must have required fields
        assert 'X' in inputs, 'Input must have X'
        assert 'y' in inputs, 'Input must have y'

        # Local variables
        X = inputs['X']
        y = inputs['y']
        data_test = inputs.get('data_test')

        # Start transform the data
        transformer = DistanceBasedTransformer(**self.config)

        X_cp = X.copy()
        if 'TARGET' in X_cp.columns:
            X_cp.drop(['TARGET'], axis=1, inplace=True)
        if 'SK_ID_CURR' in X_cp.columns:
            X_cp.drop(['SK_ID_CURR'], axis=1, inplace=True)

        X_cp = X_cp.replace(np.inf, 0)
        X_cp = X_cp.replace(-np.inf, 0)
        X_cp = X_cp.fillna(value=0)

        if data_test is not None:
            data_test_cp = data_test.copy()
            if 'TARGET' in data_test_cp.columns:
                data_test_cp.drop(['TARGET'], axis=1, inplace=True)
            if 'SK_ID_CURR' in data_test_cp.columns:
                data_test_cp.drop(['SK_ID_CURR'], axis=1, inplace=True)

            data_test_cp = data_test_cp.replace(np.inf, 0)
            data_test_cp = data_test_cp.replace(-np.inf, 0)
            data_test_cp = data_test_cp.fillna(value=0)

            X_all = np.concatenate([X_cp, data_test_cp])
        else:
            X_all = X_cp

        logger.info('---> Fitting the all data ...')
        transformer.fit(X_all)

        # Cache the result for other can retrieve
        logger.info('---> Transforming the X train data ...')
        self.X = transformer.transform(X_cp)
        self.y = y
        self.submission_ids = inputs.get('submission_ids')
        if data_test is not None:
            logger.info('---> Transforming the X test data ...')
            data_test_cp = data_test_cp.reset_index(drop=True)
            self.data_test = transformer.transform(data_test_cp)

    def get_output(self):
        return {
            'X': self.X,
            'y': self.y,
            'data_test': self.data_test,
            'submission_ids': self.submission_ids
        }


class DimensionReductionPip(BaseBlockPip):
    """ This block used for generating distance based features """

    X = None
    y = None
    data_test = None
    submission_ids = None

    config = {}

    def _execute(self, inputs):

        # Inputs must have required fields
        assert 'X' in inputs, 'Input must have X'
        assert 'y' in inputs, 'Input must have y'

        # Local variables
        X = inputs['X']
        y = inputs['y']
        data_test = inputs.get('data_test')

        # Params
        feats = X.columns.difference(['SK_ID_CURR', 'TARGET', 'index'])
        n_components = self.config['n_components']
        X_copy = X[['SK_ID_CURR']].copy()
        data_test_copy = data_test[['SK_ID_CURR']].copy()

        # Reduce dimension by PCA
        # Train data
        decomposition_clf = PCA(n_components=n_components)
        X = X[feats]
        X = X.replace(np.inf, 0)
        X = X.replace(-np.inf, 0)
        X = X.fillna(value=0)

        decomposition_clf.fit(X)
        X_train_decomposition = decomposition_clf.transform(X)

        # Test data
        decomposition_clf = PCA(n_components=n_components)
        data_test = data_test[feats]
        data_test = data_test.replace(np.inf, 0)
        data_test = data_test.replace(-np.inf, 0)
        data_test = data_test.fillna(value=0)

        decomposition_clf.fit(data_test)
        data_test_decomposition = decomposition_clf.transform(data_test)

        try:
            self.X = pd.concat([X_copy, pd.DataFrame(X_train_decomposition)], axis=1, sort=False)
        except:
            self.X = pd.concat([X_copy, pd.DataFrame(X_train_decomposition)], axis=1)

        try:
            self.data_test = pd.concat([data_test_copy, pd.DataFrame(data_test_decomposition)], axis=1, sort=False)
        except:
            self.data_test = pd.concat([data_test_copy, pd.DataFrame(data_test_decomposition)], axis=1)
        self.y = y
        self.submission_ids = inputs.get('submission_ids')

    def get_output(self):
        return {
            'X': self.X,
            'y': self.y,
            'data_test': self.data_test,
            'submission_ids': self.submission_ids
        }
