# -*- coding: utf-8 -*-
# import tempfile
import gc
import os

from sklearn.externals import joblib
from sklearn.base import BaseEstimator, ClassifierMixin

from csef.session import SessionManager


class OOFBaseClassifier(BaseEstimator, ClassifierMixin):
    """docstring for OOFBaseClassifier"""

    clf_class = None
    release_after_fit = False
    features = []
    is_fitted = False

    def __init__(
            self, model_name, fold, **kwargs):
        """Init"""
        self.kwargs = kwargs
        self.model_name = model_name
        self.fold = fold
        self.clf = self.clf_class(**kwargs)

    @property
    def feature_importances(self):
        """Feature Importance"""
        if hasattr(self.clf, 'feature_importances_'):
            return self.clf.feature_importances_
        else:
            return []

    @property
    def evals_result(self):
        """Evaluate Result"""
        if hasattr(self.clf, 'evals_result_'):
            return self.clf.evals_result_
        else:
            return []

    def _get_save_model_path(self):
        model_name = "pipeline.{}.fold{}.pkl".format(self.model_name, self.fold)
        return os.path.join(os.environ['PROJ_HOME'], 'models', str(SessionManager().session_id), model_name)

    def normalize_eval_metric(self, eval_metric):
        """
        This method helps to add the error metric to the fit

        :param eval_metric: The string or list of string
        :return: The new eval_metric
        """
        if not type(eval_metric) == list:
            eval_metric = [eval_metric]

        if 'error' not in eval_metric:
            eval_metric.append('error')

        return eval_metric

    def fit(self, X, y=None, eval_set=None, eval_metric=None, verbose=False, early_stopping_rounds=100):
        """Fit"""
        # Add error metric for validation
        if eval_metric:
            eval_metric = self.normalize_eval_metric(eval_metric)

        self.clf.fit(
            X, y,
            eval_set=eval_set,
            eval_metric=eval_metric, verbose=verbose, early_stopping_rounds=early_stopping_rounds
        )

        self.is_fitted = True

        return self

    def predict(self, X):
        """Predict"""
        return self.clf.predict(X)

    def predict_proba(self, X):
        """Predict Probability"""
        if not hasattr(self, "clf"):
            self.load_model()
        return self.clf.predict_proba(X)

    def save_model(self):
        """Save model"""
        if hasattr(self, "clf"):
            joblib.dump(self.clf, self._get_save_model_path())

    def load_model(self):
        """Load model"""
        self.clf = joblib.load(self._get_save_model_path())
        return self

    def release_resource(self):
        """Release resouce"""
        self.save_model()

        # Release memory
        del self.clf
        gc.collect()

        return self

    def get_fit_stats(self):
        """
        Get some fit stats for recording. Example: best_params, evals_result ..

        Sub class can extend to return itself stats
        :return: The dict
        """

        if not self.is_fitted:
            return {}

        if not hasattr(self, "clf"):
            self.load_model()

        return {
            'evals_result': self.evals_result,
            'feature_importances': self.feature_importances
        }
