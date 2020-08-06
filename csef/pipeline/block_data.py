# -*- coding: utf-8 -*-
import gc
import os
import time
import pandas as pd

from csef.pipeline import PipelineStorageManager
from csef.utils.logging import getLogger
from csef.pipeline.base import BaseBlockPip
from csef.session import SessionManager
# from csef.data.load_data import load_processed_data, load_x_y, _get_config_file_path
from csef.utils.helper import get_proj_home

logger = getLogger(logger_name=__name__)


class DataLoadingBlockPip(BaseBlockPip):
    """This block used for load data trained by another process or from the DataLoadingBlockPip"""

    X = None
    y = None
    data_test = None
    submission_ids = None
    config = {
        'target': 'TARGET'
    }

    def _execute(self, inputs):

        # Get global variable from session manager
        proj_home = get_proj_home()
        data_version = SessionManager().get_prop('data_version')
        data_tag = SessionManager().get_prop('data_tag')
        data_extension = SessionManager().get_prop('data_extension', 'csv')
        make_submission = SessionManager().get_prop('make_submission')
        sample = SessionManager().get_prop('sample')
        seed = SessionManager().get_prop('seed')

        if data_tag == 'stable':
            train_local_path = '{proj_home}/data/processed/{data_version}/application_train.{data_extension}'\
                .format(
                    proj_home=proj_home,
                    data_version=data_version,
                    data_extension=data_extension
                )
            test_local_path = '{proj_home}/data/processed/{data_version}/application_test.{data_extension}'\
                .format(
                    proj_home=proj_home,
                    data_version=data_version,
                    data_extension=data_extension
                )
        else:
            train_local_path = '{proj_home}/data/processed/{data_version}/{data_tag}/application_train.{data_extension}'\
                .format(
                    proj_home=proj_home,
                    data_version=data_version,
                    data_extension=data_extension,
                    data_tag=data_tag
                )
            test_local_path = '{proj_home}/data/processed/{data_version}/{data_tag}/application_test.{data_extension}'\
                .format(
                    proj_home=proj_home,
                    data_version=data_version,
                    data_extension=data_extension,
                    data_tag=data_tag
                )

        # Try to download the data files if not existing
        if not os.path.isfile(train_local_path):
            logger.info('---> Downloading data ... ')
            PipelineStorageManager().sync_data_files(data_version, data_tag)

            # Google api has rate condition and make the thread async
            # so need to implement this code to make sure we have data before
            sleep_counter = 0
            while not os.path.isfile(train_local_path):
                time.sleep(6)
                sleep_counter += 1
                if sleep_counter > 10:
                    raise Exception("Had an issue when download data!")

        logger.info('---> Loading data {} ... '.format(train_local_path))
        data_train = load_processed_data(train_local_path, is_full_path=True)

        if make_submission:
            logger.info('---> Loading data {} ... '.format(test_local_path))
            self.data_test = load_processed_data(test_local_path, is_full_path=True)
            self.submission_ids = self.data_test['SK_ID_CURR']

        # If sample provided, need to get the sample instead of train the data with the full data
        if sample < 1:
            logger.info('---> Sample data with fraction: {} ... '.format(sample))
            data_train = data_train.sample(frac=float(sample), random_state=seed)

        # Get the target and data for training
        self.X, self.y = load_x_y(data_train, self.config['target'])

        # Clean up the data
        del data_train
        gc.collect()

    def get_output(self):
        """Return output"""
        return {
            'X': self.X,
            'y': self.y,
            'data_test': self.data_test,
            'submission_ids': self.submission_ids
        }


class LoadProcessedDataBlockPip(BaseBlockPip):
    """
    This block is used to load processed data from raw.
    """

    train = None
    test = None

    def _execute(self, inputs):
        proj_home = get_proj_home()

        # locate path
        train_data_path = f'{proj_home}/../data/processed/train.gzip'
        test_data_path = f'{proj_home}/../data/processed/test.gzip'

        # load data
        train_data = pd.read_pickle(
            train_data_path, compression='gzip')
        test_data = pd.read_pickle(
            test_data_path, compression='gzip')

        self.train = train_data
        self.test = test_data

    def get_output(self):
        """Return output"""
        return {
            'train': self.train,
            'test': self.test
        }
