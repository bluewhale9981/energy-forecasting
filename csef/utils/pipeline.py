# -*- coding: utf-8 -*-
from termcolor import colored

from csef.utils.helper import datetime_diff


def get_status(result):
    """Get status of a result"""
    status = colored('RUNNING', 'yellow')

    is_error = result['is_error'] if 'is_error' in result else False
    is_finished = result['is_finished']

    if is_finished and not is_error:
        status = colored('DONE', 'green')
    elif is_error:
        status = colored('FAIL', 'red')

    return status


def get_time_taken(result):
    """Get time taken of a result"""
    thour, tmin, tsec = datetime_diff(result['created'], result['finished_on'])
    return '%i:%i:%i' % (thour, tmin, round(tsec, 0))
