# -*- coding: utf-8 -*-
import time
import os

from csef.utils.design_patterns import SingletonDecorator


@SingletonDecorator
class SessionManager(object):

    session_id = int(time.time())
    _props = {
        'config_file': None,
        'seed': 100,
        'debug': True,
        'make_submission': False,
        'sample': 1,
        'remote_log': False,
        'remote_result': False
    }

    def renew(self, props, session_id=None):
        """Review session with the props and new session id"""
        self.session_id = session_id if session_id else int(time.time())
        self._props.update(props)

    def get_props(self):
        props = self._props.copy()
        props.update({
            'id': self.session_id
        })
        return props

    def get_prop(self, prop_name, default=None):
        """Get the prop from """
        return self._props.get(prop_name, default)

    def get_prop_normalize_config_name(self):
        """ Get the normalized config name """
        config_file = self._props.get('config_file')

        return os.path.splitext(os.path.basename(config_file))[0]
