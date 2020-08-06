# -*- coding: utf-8 -*-
from __future__ import absolute_import

import platform
import sys

import pandas as pd
import numpy as np

import matplotlib

import sklearn


class EnvironmentChecker(object):
    """
    Show the environment information.
    """
    def __init__(self):
        pass

    def get_info(self):
        """
        Method to get the environment information in the current system.
        """
        print('Operating system version....', platform.platform())
        print('Python version is........... %s.%s.%s' % sys.version_info[:3])
        print('scikit-learn version is.....', sklearn.__version__)
        print('pandas version is...........', pd.__version__)
        print('numpy version is............', np.__version__)
        print('matplotlib version is.......', matplotlib.__version__)
