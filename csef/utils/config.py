import os

from csef.utils.helper import load_config


def load_default_config():
    """
    Load default config from `pipeline-configs/default.yml`
    """
    proj_home = os.environ.get('PROJ_HOME', '')
    return load_config(os.path.join(proj_home, 'pipeline-configs/default.yml'))
