# -*- coding: utf-8 -*-
import yaml
import json
import os
from datetime import datetime
import traceback

import pandas as pd

from os import listdir
from os.path import isfile, join

from csef.utils.helper import load_class, dir_init, get_proj_home, dict_deep_update
from csef.utils.git import get_commit_id, get_global_username
from csef.utils.design_patterns import SingletonDecorator
from csef.utils.google_datastore import GoogleDataStore
from csef.utils.google_storage import GoogleStorage
from csef.utils.logging import getLogger
from csef.session import SessionManager

from csef.utils.ensembles import generate_submission_avg, generate_submission_geomean, generate_submission_rankavg, generate_submission_vote


logger = getLogger(logger_name=__name__)


def load_config(config_file):
    """
    Load the config file in yml format

    :param config_file: The path to config file
    :return: The dict contains configs
    """
    with open(config_file) as f:
        return yaml.load(f)


@SingletonDecorator
class PipelineRecorder(object):
    """
    PipelineRecorder
    """

    _cache = {}

    session_id = None
    config_file = None
    remote_log = False

    @property
    def session_id(self):
        return SessionManager().session_id

    @property
    def config_file(self):
        return SessionManager().get_prop('config_file')

    @property
    def remote_log(self):
        return SessionManager().get_prop('remote_log')

    def record(self, data):
        """
        Record the data for current session.
        :param data: The data need to recorded
        :return: Self
        """

        # Init the session when start the recording
        if self.session_id not in self._cache:
            self._cache[self.session_id] = {}

        self._cache[self.session_id] = dict_deep_update(self._cache[self.session_id], data)

        return self

    def push(self):
        """
        Push the current data to remote storage (Google datastore).
        :return: Self
        """
        GoogleDataStore().upsert_pipeline_result(self._cache[self.session_id])

        return self

    def record_and_push(self, data):
        """
        Record the data for current session and push
        :param data: The data need to recorded
        :return: Self
        """
        return self.record(data).push()

    def pull_session(self, session_id):
        """
        Pull data for a session from the remote storage
        :param session_id: The session id need to pull data.
        :return: Self
        """
        self._cache[session_id] = GoogleDataStore().get_result_by_session(session_id)
        return self

    def get_session(self, session_id):
        """
        Get the data by session id
        :param session_id: The session need to get
        :return:
        """
        return self._cache[session_id]

    def get_current_session(self):
        """
        Get current session
        :return: The dict data of session
        """
        return self.get_session(self.session_id)

    def clean_session(self, session_id):
        """
        Clean a session by session id
        :param session_id: The session need to clean
        :return: Self
        """
        self._cache[session_id] = {}
        return self

    def clean_current_session(self):
        """
        Clean a session by session id
        :param session_id: The session need to clean
        :return: Self
        """
        return self.clean_session(self.session_id)

    def dump_stats(self, key, data):
        """
        Some stats too big and can not save to google datastore, so need to save to json file to read later
        :param data: The data need to save
        :param key: The key quick be used as name for the file.
        """
        file_path = os.path.join(os.environ['PROJ_HOME'], 'models', str(self.session_id), key + '.json')

        # Some data can be
        data = GoogleDataStore().normalize_data(data)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True, default=str)


@SingletonDecorator
class PipelineStorageManager(object):

    session_id = None
    config_file = None
    remote_log = False

    # Assume this bucket created by console browser
    # TODO: think about how to reuse this for another project
    bucket_name = 'ml-hcdr-bucket'

    @property
    def session_id(self):
        return SessionManager().session_id

    @property
    def config_file(self):
        return SessionManager().get_prop('config_file')

    @property
    def remote_log(self):
        return SessionManager().get_prop('remote_log')

    def _generate_data_folder(self, version, tag):
        tag = ('/' + tag) if tag != 'stable' else ''
        return "data/processed/{version}{tag}".format(version=version, tag=tag)

    def generate_submission_filename(self, config_file, data_version, session_id, tag='stable'):
        return 'pipeline-{config_file}.{data_version}.{tag}.{session_id}.csv'.format(
            config_file=config_file,
            data_version=data_version,
            tag=tag,
            session_id=session_id
        )

    def upload_submission_file(self, config_file, data_version, session_id, tag='stable'):
        """Upload the submission file to google storage"""

        submission_file_name = self \
            .generate_submission_filename(config_file, data_version, session_id, tag)
        submission_file_name = 'submissions/{}'.format(submission_file_name)
        source_file_name = os.path.join(os.environ['PROJ_HOME'], submission_file_name)

        GoogleStorage().upload_blob(self.bucket_name, source_file_name, submission_file_name)
        print('Uploaded submission file {}'.format(source_file_name))

    def sync_submission_files(self):
        """ Download all submission file to local """
        blobs = GoogleStorage().list_blobs_with_prefix(self.bucket_name, 'submissions')

        for blob in blobs:
            destination_file_name = os.path.join(os.environ['PROJ_HOME'], blob.name)

            # Check if the local file exist before download file
            if not os.path.isfile(destination_file_name):
                blob.download_to_filename(destination_file_name)
                print('Downloaded file {destination_file_name}'.format(destination_file_name=destination_file_name))

    def sync_files(self, folder):
        """Sync file from storage to local"""
        blobs = GoogleStorage().list_blobs_with_prefix(self.bucket_name, folder)

        # Create the session folder if not existing
        project_home = os.environ['PROJ_HOME']
        root_folder = os.path.join(project_home, folder)
        if not os.path.isdir(root_folder):
            os.makedirs(root_folder)

        # Start download files
        for blob in blobs:
            destination_file_name = os.path.join(project_home, blob.name)

            # Check if the local file exist before download file
            if not os.path.isfile(destination_file_name):

                # Create folder to avoid exception when download
                destination_file_folder = os.path.dirname(destination_file_name)
                if not os.path.isdir(destination_file_folder):
                    os.makedirs(destination_file_folder)

                blob.download_to_filename(destination_file_name)
                print('Downloaded file {}'.format(destination_file_name))

    def upload_files(self, folder):
        """Upload file from local to storage"""

        # Load all blobs in the session to make sure only upload needed files
        blobs = GoogleStorage().list_blobs_with_prefix(self.bucket_name, folder)
        blobs = [blob.name for blob in blobs]

        project_home = os.environ['PROJ_HOME']
        root_folder = os.path.join(project_home, folder)

        for file in os.listdir(root_folder):
            file_name = "{folder}/{file}".format(folder=folder, file=file)
            if file_name not in blobs:
                source_file_name = os.path.join(project_home, file_name)
                GoogleStorage().upload_blob(
                    self.bucket_name, source_file_name, file_name)
                print('Uploaded file {}'.format(source_file_name))

    def upload_training_files(self, session_id):
        """Upload training files (in models folder) of a session id"""
        session_id = str(session_id)
        folder = 'models/{}'.format(session_id)
        self.upload_files(folder)

    def sync_training_files(self, session_id):
        """Download training file from google storage"""
        session_id = str(session_id)
        folder = 'models/{}'.format(session_id)
        self.sync_files(folder)

    def upload_data_files(self, version, tag='stable', compress=False):
        """Upload data file to google storage"""
        data_folder = self._generate_data_folder(version, tag)

        # Try to compress the data before upload
        if compress:
            project_home = os.environ['PROJ_HOME']
            root_folder = os.path.join(project_home, data_folder)

            train_local_path = "{root_folder}/application_train.csv".format(root_folder=root_folder)
            test_local_path = "{root_folder}/application_test.csv".format(root_folder=root_folder)

            if os.path.isfile(train_local_path):
                print('Compressing the data files...')
                train_df = pd.read_csv(train_local_path)
                train_df.to_pickle(
                    '{root_folder}/application_train.gzip'.format(root_folder=root_folder), compression='gzip')
                test_df = pd.read_csv(test_local_path)
                test_df.to_pickle(
                    '{root_folder}/application_test.gzip'.format(root_folder=root_folder), compression='gzip')

                # Remove the csv files
                os.remove(train_local_path)
                os.remove(test_local_path)

        self.upload_files(data_folder)

    def sync_data_files(self, version, tag='stable'):
        """Download data file from google storage"""
        data_folder = self._generate_data_folder(version, tag)
        self.sync_files(data_folder)

    def generate_ensemble_submission_filename(self, kind_of_ensemble):
        return 'pipeline-{}-ensemble-{}.csv'.format(kind_of_ensemble, self.session_id)

    def reformat_submission_files(self, submission_local_path=None):
        """ Re-format the values of the submission files at locally if need """
        if submission_local_path is not None:
            sub_files = [f for f in listdir(submission_local_path) if isfile(join(submission_local_path, f))]
            for sub_file in sub_files:
                submission_local_file_path = '{}/{}'.format(submission_local_path, sub_file)
                sub_df = pd.read_csv(submission_local_file_path)
                if ('SK_ID_CURR' in sub_df.columns) and (sub_df.shape[0] == 48744):
                    sub_df['SK_ID_CURR'] = sub_df['SK_ID_CURR'].astype(int)
                    sub_df.to_csv(submission_local_file_path, index=False)
                else:
                    os.remove(submission_local_file_path)

    def ensembling_submission_files(self, kind_of_ensembles_supported=None):
        if kind_of_ensembles_supported is None:
            kind_of_ensembles_supported = ['vote', 'vote_weighted',
                                           'rankavg', 'avg', 'geomean']

            # Save ensembled submissions file in `ensemble-submissions` folder at root project
            ensemble_submission_local_path = os.path.join(os.environ['PROJ_HOME'], 'ensemble-submissions')

            # Create directory if doesn't exist
            # Delete the contents of a current directory
            dir_init(ensemble_submission_local_path)

            for kind_of_ensemble in kind_of_ensembles_supported:
                ensemble_submission_file_name = self \
                    .generate_ensemble_submission_filename(kind_of_ensemble)
                ensemble_submission_file_name = 'ensemble-submissions/{}'.format(ensemble_submission_file_name)

                if kind_of_ensemble == 'vote':
                    generate_submission_vote('./submissions/*.csv', ensemble_submission_file_name)
                elif kind_of_ensemble == 'vote_weighted':
                    # FIXME: Current this approach same the result with `vote`
                    generate_submission_vote('./submissions/*.csv', ensemble_submission_file_name, weights='weighted')
                elif kind_of_ensemble == 'rankavg':
                    generate_submission_rankavg('./submissions/*.csv', ensemble_submission_file_name)
                elif kind_of_ensemble == 'avg':
                    generate_submission_avg('./submissions/*.csv', ensemble_submission_file_name)
                elif kind_of_ensemble == 'geomean':
                    generate_submission_geomean('./submissions/*.csv', ensemble_submission_file_name)

        else:
            print('{} does not support now!'.format(kind_of_ensemble))

    def sync_and_ensemble_top_submission_files(self, sids):
        """
        Download all top submission file to local and
        do ensemble these submission file based on session id list
        """

        # Download all submission file if sids is empty array
        if len(sids) < 1:
            self.sync_submission_files()
        else:
            blobs = GoogleStorage().list_blobs_with_prefix(self.bucket_name, 'submissions')
            submission_local_path = os.path.join(os.environ['PROJ_HOME'], 'submissions')

            # Create directory if doesn't exist
            # Delete the contents of a current directory
            dir_init(submission_local_path)

            for blob in blobs:
                for sid in sids:
                    session_id = str(sid)
                    # Hard code to download only stable version
                    if (session_id in blob.name) and ('stable' in blob.name):
                        destination_file_name = os.path.join(os.environ['PROJ_HOME'], blob.name)

                        # Check if the local file exist before download file
                        if not os.path.isfile(destination_file_name):
                            blob.download_to_filename(destination_file_name)
                            print('Downloaded file {}'.format(destination_file_name))

                        # Break out from the inside loop
                        break

            # Re-format the submission files before blending
            self.reformat_submission_files(submission_local_path)

            # Do blending
            self.ensembling_submission_files()


class PipelineManager(object):
    """Manager for all blocks of the pipeline"""

    config = None
    args = None
    running = False
    finished = False
    is_error = False
    _blocks = []
    _current_idx = 0

    def _get_inputs(self, input_keys):

        if not input_keys:
            return None

        outputs = {}

        # The input_keys can be list or single string
        input_keys_list = input_keys if type(input_keys) == list else [input_keys]

        for block in self._blocks:
            if block.name in input_keys_list and block.executed:
                outputs[block.name] = block.get_output()

        return outputs if len(input_keys_list) > 1 else outputs[input_keys]

    def init(self, config, args):
        """
        The class manage the pipeline with config and args.
        :param config: The dict of config loaded in the file.
        :param args: The arg custom for the pipeline running.
        """
        self.config = config
        self.args = args

        assert config['pipeline']
        assert config['metadata']

        # Renew to start a session
        args.update({
            'commit_id': get_commit_id(),
            'owner': get_global_username()
        })
        SessionManager().renew(args)
        session_id = SessionManager().session_id

        logger.info("### Start the pipeline session id {}".format(session_id))
        logger.info("### {}".format(args))

        # Starting record the session
        pipeline_data = SessionManager().get_props()
        pipeline_data.update({
            'created': datetime.now(),
            'metadata': config['metadata'],
            'is_finished': False
        })
        PipelineRecorder().record_and_push(pipeline_data)

        self.running = False
        self.finished = False
        self.is_error = False

        # Build the block of pipeline
        self._blocks = []
        self._current_idx = 0

        pipeline_block_definitions = config['pipeline']

        for pipeline_block_definition in pipeline_block_definitions:
            class_name = pipeline_block_definition['class_name']
            clazz = load_class(class_name)

            # Check if there is config_from_file to load the config from the file
            if 'config_from_file' in pipeline_block_definition:
                config_from_file = os.path.join(get_proj_home(), pipeline_block_definition['config_from_file'])
                block_config = load_config(config_from_file)
            else:
                block_config = pipeline_block_definition.get('config', {})

            inputs_from = pipeline_block_definition.get('inputs_from')
            block_instance = clazz(pipeline_block_definition['name'], block_config, self, inputs_from)

            self._blocks.append(block_instance)

        return self

    def run(self):
        """Start running the pipeline"""
        self.running = True

        try:
            for block in self._blocks:
                inputs = self._get_inputs(block.inputs_from)
                block.execute(inputs)

            self.is_error = False
        except:
            error_message = traceback.format_exc()

            print(error_message)

            error_message = (error_message[:1400] + '..') if len(error_message) > 1400 else error_message
            logger.error(error_message)

            # Try to log error message
            self.is_error = True
            PipelineRecorder().record({
                'error': error_message
            })

        return self

    def finish(self):
        """Finish the pipeline"""

        self.finished = True
        self.running = False

        # Finish the session
        PipelineRecorder().record_and_push({
            'is_finished': True,
            'is_error': self.is_error,
            'finished_on': datetime.now()
        })

        logger.info("### Finished the pipeline session id {}".format(SessionManager().session_id))

        return self
