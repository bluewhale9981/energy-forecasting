# -*- coding: utf-8 -*-
from __future__ import absolute_import

"""Collection of ML logging utilities."""

import time
from hashids import Hashids
import logging

from csef.utils.google_datastore import GoogleDataStore
from csef.session import SessionManager


logging.basicConfig(level=logging.INFO)


def getLogger(level=logging.INFO, logger_name=__name__, version=1):
    """Build the logger and set the logger module

    Refer to logging.*

    Parameters
        :param level: optional, default: logging.INFO
        :type level: int

        :param logger_name: optional. The name of logger, default: __name__
        :type logger_name: str

        :param version: optional. The version number of the logging, default: 1
        :type version: int

    Returns
        src.utils.logging.MLLogger
    """
    return MLLogger(level, logger_name, version)


class MLLogger(object):
    """
    The logger class for logging training results.

    Parameters
        :param level: optional, default: logging.INFO
        :type level: int

        :param logger_name: optional. The name of logger, default: __name__
        :type logger_name: str

        :param version: optional. The version number of the logging, default: 1
        :type version: int
    """

    hashids = Hashids(min_length=16)

    def __init__(self, level=logging.INFO, logger_name=__name__, version=1):
        self.session_id = self.hashids.encode(SessionManager().session_id)

        logger = logging.getLogger(logger_name + '#' + self.session_id)
        logger.setLevel(level)

        self.logger = logger

        self.extra = {
            'session_id': self.session_id
        }

    @property
    def remote_log(self):
        return SessionManager().get_prop('remote_log')

    def log(self, level, message, extra={}):
        """
        Log the message with extra information.

        Parameters
            :param level: optional, default: logging.INFO
            :type level: int

            :param message: optional. The name of logger, default: __name__
            :type message: str

            :param extra: The extra information need to send to log endpoint, default: {}
            :type extra: dict
        """
        self.logger.log(level, message, extra=dict(self.extra, **extra))

        # Log on the cloud
        if self.remote_log:
            GoogleDataStore().upsert_pipeline_log({
                'message': message,
                'level': level
            })

    def error(self, message, extra={}):
        """
        Log the error message with extra information.

        Parameters
            :param level: optional, default: logging.INFO
            :type level: int

            :param message: optional. The name of logger, default: __name__
            :type message: str

            :param extra: The extra information need to send to log endpoint, default: {}
            :type extra: dict
        """
        self.log(logging.ERROR, message, extra=extra)

    def info(self, message, extra={}):
        """
        Log the info message with extra information.

        Parameters
            :param level: optional, default: logging.INFO
            :type level: int

            :param message: optional. The name of logger, default: __name__
            :type message: str

            :param extra: The extra information need to send to log endpoint, default: {}
            :type extra: dict
        """
        self.log(logging.INFO, message, extra=extra)

    def warning(self, message, extra={}):
        """
        Log the warning message with extra information.

        Parameters
            :param level: optional, default: logging.INFO
            :type level: int

            :param message: optional. The name of logger, default: __name__
            :type message: str

            :param extra: The extra information need to send to log endpoint, default: {}
            :type extra: dict
        """
        self.log(logging.WARNING, message, extra=extra)

    def debug(self, message, extra={}):
        """
        Log the debug message with extra information.

        Parameters
            :param level: optional, default: logging.INFO
            :type level: int

            :param message: optional. The name of logger, default: __name__
            :type message: str

            :param extra: The extra information need to send to log endpoint, default: {}
            :type extra: dict
        """

        self.log(logging.DEBUG, message, extra=extra)
