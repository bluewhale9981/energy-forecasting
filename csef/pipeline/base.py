# -*- coding: utf-8 -*-
import yaml

from csef.utils.logging import getLogger


logger = getLogger(logger_name=__name__)


def load_config(config_file):
    """
    Load the config file in yml format
    :param config_file: The path to config file
    :return: The dict contains configs
    """
    with open(config_file) as f:
        return yaml.load(f)


class BaseBlockPip(object):

    executed = False
    config = {}

    def __init__(self, name, config, pip_manager, inputs_from=None):
        self.name = name
        self.pip_manager = pip_manager
        self.inputs_from = inputs_from

        # Use the default config and update with the config from params
        if type(config) == dict:
            self.config.update(config)
        else:
            self.config = config

    def _execute(self, inputs):
        raise Exception('This method must be overridden!')

    def get_output(self):
        raise Exception('This method must be overridden!')

    def execute(self, inputs=None):
        logger.info('###### Executing the block: {} ...'.format(self.name))
        self._execute(inputs)
        self.executed = True
        logger.info('###### Finished the block: {}!'.format(self.name))

    def clean(self):
        pass
