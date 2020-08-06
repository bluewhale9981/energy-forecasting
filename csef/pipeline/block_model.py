# -*- coding: utf-8 -*-
import os

import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from csef.pipeline import PipelineStorageManager
from csef.session import SessionManager
from csef.utils.logging import getLogger
from csef.pipeline.base import BaseBlockPip
from csef.utils.helper import load_class, timer, get_proj_home
import csef.utils.naming as namingUtils

logger = getLogger(logger_name=__name__)


class ModelTrainingBlockPip(BaseBlockPip):
    """ This block used for training model """

    pipeline = None

    def _build_pipeline(self):
        """
        Build the pipeline based on config file. This method help to define a complex
        pipeline
        :return: The Pipeline
        """

        def traverse_obj(obj, path=None):
            """
            Deep traverse and processing the value of a dict
            :param obj: The python dict need to processed
            :param path: The path
            :return: The processed dict
            """
            if path is None:
                path = []
            if isinstance(obj, dict):
                if 'class_name' in obj:

                    assert obj['params'], 'The class definition must have params'

                    params = traverse_obj(obj['params'])
                    instance = load_class(obj['class_name'])(**params)

                    value = (obj['name'], instance) if 'name' in obj else instance
                else:
                    value = {k: traverse_obj(v, path + [k])
                             for k, v in obj.items()}
            elif isinstance(obj, list):
                value = [traverse_obj(elem, path + [[]])
                         for elem in obj]
            else:
                value = obj
            return value

        config = traverse_obj(self.config)

        return Pipeline(config)

    def _get_categorical_feature(self):
        """
        Get a list of categorical features.
        """
        try:
            config = self.config[0]
            categorical_feature = config['params']['categorical_feature']
        except:
            categorical_feature = 'auto'

        return categorical_feature

    def _execute(self, inputs):

        # Inputs must have required fields
        assert 'X' in inputs, 'Input must have X'
        assert 'y' in inputs, 'Input must have y'

        # Local variables
        X = inputs['X']
        y = inputs['y']
        categorical_feature = inputs.get('categorical_feature', self._get_categorical_feature())

        data_test = inputs.get('data_test')
        make_submission = SessionManager().get_prop('make_submission')
        dump_pipeline = SessionManager().get_prop('dump_pipeline')
        session_id = SessionManager().session_id
        normalize_config_name = SessionManager().get_prop_normalize_config_name()
        config_file = SessionManager().get_prop('config_file')
        data_version = SessionManager().get_prop('data_version')
        data_tag = SessionManager().get_prop('data_tag')

        pipeline = self.pipeline = self._build_pipeline()

        # Save the pipeline
        session_folder = namingUtils.get_session_folder(session_id)
        pipeline_file_name = namingUtils.get_pipeline_name(config_file, data_version, session_id)

        # Start training process
        start_time = timer()

        # Set categorical feature
        named_steps = pipeline.named_steps

        for key, value in named_steps.items():
            if key == 'OutOfFoldClassifier':
                if hasattr(named_steps[key], 'set_categorical_feature'):
                    outOfFoldClassifier = named_steps[key]
                    outOfFoldClassifier.set_categorical_feature(categorical_feature)

        pipeline.fit(X, y)
        timer(start_time)

        # Save the pipeline
        if dump_pipeline:
            pipeline_local_path = "{}/{}".format(session_folder, pipeline_file_name)
            joblib.dump(pipeline, pipeline_local_path)
            logger.info('---> Saved pipeline file at {}'.format(pipeline_local_path))

            # Upload model files to google storage
            PipelineStorageManager().upload_training_files(session_id)

        if make_submission:
            assert 'submission_ids' in inputs, 'Input must have submission_ids'
            submission_ids = inputs['submission_ids']

            submission = pd.DataFrame()
            submission['SK_ID_CURR'] = submission_ids.astype(int)

            if 'TARGET' in data_test.columns:
                data_test.drop(['TARGET'], axis=1, inplace=True)
            if 'SK_ID_CURR' in data_test.columns:
                data_test.drop(['SK_ID_CURR'], axis=1, inplace=True)

            submission['TARGET'] = pipeline.predict_proba(data_test)[:, 1]

            submission_local_path = os.path.join(get_proj_home(), 'submissions')
            if not os.path.isdir(submission_local_path):
                os.mkdir(submission_local_path)

            submission_file_name = PipelineStorageManager() \
                .generate_submission_filename(normalize_config_name, data_version, session_id, data_tag)
            submission_local_path = os.path.join(submission_local_path, submission_file_name)

            submission.to_csv(submission_local_path, index=False, float_format='%8f')
            logger.info('Saved submission file at {}'.format(submission_local_path))

            # Upload submission file to google storage
            PipelineStorageManager().upload_submission_file(normalize_config_name, data_version, session_id, data_tag)

    def get_output(self):
        return {
            'pipeline': self.pipeline
        }


class ModelTuningBlockPip(BaseBlockPip):
    """ This block used for tuning model """

    pass
