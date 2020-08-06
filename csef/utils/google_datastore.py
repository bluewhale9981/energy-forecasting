# -*- coding: utf-8 -*-
import os
from datetime import datetime

from google.cloud import datastore
import numpy as np
from pandas.core.indexes.base import Index

from csef.session import SessionManager
from csef.utils.design_patterns import SingletonDecorator


@SingletonDecorator
class GoogleDataStore(object):

    def __init__(self):
        """
        This class defines the wrapper for google datastore.
        """
        self.client = datastore.Client()

    def _update_default_data(self, data):
        default = {
            'created': datetime.now(),
            'session_id': self.session_id,
            'config_file': self.config_file
        }

        # In case we already has the created in the data
        # we need to leave the created here to keep the timeline of process
        if 'created' in data:
            del default['created']

        default.update(data)

        return default

    def _build_pipeline_result_key(self, session_id):
        return (
            'Pipeline', self.config_file,
            'Result', session_id
        )

    @property
    def session_id(self):
        return SessionManager().session_id

    @property
    def config_file(self):
        return SessionManager().get_prop('config_file')

    @property
    def remote_result(self):
        return SessionManager().get_prop('remote_result')

    def normalize_data(self, data, path=None):
        """
        Normalize data, convert
        :param data:
        :param path:
        :return:
        """
        if path is None:
            path = []
        if isinstance(data, dict):
            value = {k: self.normalize_data(v, path + [k])
                     for k, v in data.items()}
        elif isinstance(data, list):
            value = [self.normalize_data(elem, path + [[]])
                     for elem in data]
        elif isinstance(data, np.ndarray) or isinstance(data, Index):
            value = data.tolist()
        else:
            value = data
        return value

    def upsert(self, key, data):
        """
        Update and insert with key
        :param key: The key in tuple type
        :param data: The data
        """

        # Doesn't allow if the session flag set to false
        if not self.remote_result:
            return

        pipeline_key = self.client.key(*key)

        entity = datastore.Entity(key=pipeline_key)
        entity.update(self.normalize_data(data))

        self.client.put(entity)

    def upsert_pipeline_log(self, log_data):
        """
        Upsert the pipeline log
        :param log_data: The log data
        """
        key = ('Pipeline', self.config_file, 'LogList', self.session_id, 'Log')
        self.upsert(key, self._update_default_data(log_data))

    def upsert_pipeline_result(self, result_data):
        """
        Upsert the pipeline result
        :param result_data: The result data
        """
        key = self._build_pipeline_result_key(self.session_id)
        self.upsert(key, self._update_default_data(result_data))

    def query(self, kind, filter_rules, orders=[], limit=100):
        """
        Query the datastore
        :param kind: The kind of log need to query
        :param filter_rules: filter rule in array of tuple
        :param orders: The order rules
        :param limit: The limit number of records return when do query. Default is 100
        :return: List of results
        """

        # Doesn't allow if the session flag set to false
        if not self.remote_result:
            return

        query = self.client.query(kind=kind)

        for filter_rule in filter_rules:
            query.add_filter(*filter_rule)

        if orders:
            query.order = orders

        fetch_params = {
            'limit': limit
        }

        return list(query.fetch(**fetch_params))

    def query_log(self, filter_rules, orders=[], limit=100):
        """
        Query the log
        :param filter_rules: filter rule in array of tuple
        :param orders: The order rules
        :param limit: The limit number of records return when do query. Default is 100
        :return: List of results
        """
        return self.query('Log', filter_rules, orders, limit)

    def query_result(self, filter_rules=[], orders=[], limit=100):
        """
        Query the result
        :param filter_rules: filter rule in array of tuple
        :param orders: The order rules
        :param limit: The limit number of records return when do query. Default is 100
        :return: List of results
        """
        return self.query('Result', filter_rules, orders, limit)

    def get(self, key):
        """
        Get an entity with key
        :param key: The tuple of key, example ('Pipeline', 'lightgbm-baseline', 'Result', '1234561')
        :return: The dict of value
        """
        # Doesn't allow if the session flag set to false
        if not self.remote_result:
            return

        storage_key = self.client.key(*key)
        return self.client.get(storage_key)

    def get_result_by_session(self, session_id):
        """
        Get an entity with key
        :param key: The tuple of key, example ('Pipeline', 'lightgbm-baseline', 'Result', '1234561')
        :param session_id: The session id
        :return: The dict of value
        """
        return self.get(self._build_pipeline_result_key(session_id))

    def delete(self, keys):
        """
        Delete one or multiple keys
        :param keys: A key object or list of keys
        """
        if type(keys) != list:
            keys = [keys]
        self.client.delete_multi(keys)

    def put(self, entity):
        """
        Update the entity
        :param entity: The updated entity
        """
        self.client.put(entity)
