# -*- coding: utf-8 -*-
import os

from csef.utils.helper import get_proj_home


def get_pipeline_name(config_file, data_version, session_id):
    """Get the file name of pipeline to dump to file"""
    normalize_config_name = os.path.splitext(os.path.basename(config_file))[0]
    return 'pipeline-{}.{}.{}.pkl'.format(normalize_config_name, data_version, session_id)


def get_session_folder(session_id):
    """Build the session folder and create the folder if not existing"""
    session_folder = os.path.join(get_proj_home(), 'models', str(session_id))
    if not os.path.isdir(session_folder):
        os.mkdir(session_folder)

    return session_folder

