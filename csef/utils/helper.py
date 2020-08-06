# -*- coding: utf-8 -*-
import hashlib
import tarfile
import importlib
import json
import operator
import collections
import os
import shutil
import yaml
from datetime import datetime


def md5sum(path, blocksize=65536):
    """
    Check sum of data file by name

    Args:
        path (str): The path to the file.
        blocksize (int): The blocksize number

    Returns:
        A string of hash value.
    """
    hash = hashlib.md5()

    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()


def compress_files(files, output):
    out = tarfile.open(output, mode='w:gz')
    try:
        for file in files:
            out.add(file)
    finally:
        out.close()


def load_class(full_class_string):
    """
    Dynamically load a class from a string.
    """

    class_data = full_class_string.split('.')
    module_path = '.'.join(class_data[:-1])
    class_str = class_data[-1]

    module = importlib.import_module(module_path)

    # Finally, we retrieve the Class
    return getattr(module, class_str)


def timer(start_time=None):
    """
    Record and print processed time.
    :param start_time: The stat time.
    :return: Nothing if there is start_time in param or return now.
    """
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def datetime_diff(start, end):
    """Diff between two datetime and return number of hours, mins and seconds"""
    start_time = start.timestamp()
    end_time = end.timestamp()

    thour, temp_sec = divmod((end - start).total_seconds(), 3600)
    tmin, tsec = divmod(temp_sec, 60)

    return thour, tmin, tsec


def get_top_n_params(tuning_local_filename, n=3):
    """
    Get top n best params from the tuning results with BayesianOptimization.

    :param n: The number of top best params. Default n = 3.
    :param tuning_local_filename: The result tuning file name.

    :return: The list of top n params. Returns None if occurs any error.
    """
    TUNING_LOCAL_PATH = '../../tuning/'
    tuning_local_filename = TUNING_LOCAL_PATH + tuning_local_filename
    with open(tuning_local_filename) as f:
        data = json.load(f)

    try:
        values = data['all']['values']
        val_dict = {v: k for v, k in enumerate(values)}

        sorted_values = sorted(val_dict.items(), key=operator.itemgetter(1), reverse=True)
        n_sorted_values = sorted_values[:n]

        top_n_params = []

        for sorted_item in n_sorted_values:
            top_n_params.append(data['all']['params'][sorted_item[0]])

        return top_n_params
    except:
        return None


def dict_deep_update(d, u):
    """
    Update the python dict with deep up strategy
    :param d: The dict need to be updated
    :param u: The update value
    :return: The new dict
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def pretty_time(datetime_obj):
    """ Print simple and pretty datetime """
    if type(datetime_obj) == datetime:
        return datetime_obj.strftime('%m-%d-%Y %H:%M:%S')
    else:
        return datetime_obj


def dir_init(dirname_path):
    """ Create directory if does not exist.
    Delete the contents of a folder if existed """

    # Create directories if doesn't exist
    if not os.path.exists(dirname_path):
        os.makedirs(dirname_path)
    else:
        # Delete the contents of a folder
        for file_name in os.listdir(dirname_path):
            file_path = os.path.join(dirname_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def get_proj_home():
    """
    Get the project home, defined in the environment
    :return: The project home
    """
    return os.environ.get('PROJ_HOME', '')


def load_config(config_file):
    """
    Load the config file in yml format
    :param config_file: The path to config file
    :return: The dict contains configs
    """
    with open(config_file) as f:
        return yaml.load(f)


def load_dict_item(d, item, default=None):
    """
    Load an item from dictionary.

    :arg d: The dictionary.
    :arg item: The item key string you want to get.
    :arg default: The default value if item is not in dict.

    :return: The value of item or default.
    """
    if type(d) is dict and item in d:
        return d[item]
    return default


def zero_transition(data, col):
    """
    Move a column on Pandas dataframe away from zero.
    """
    min = data[col].min()

    if min > 0:
        # No need to do anything.
        return data[col]
    elif min == 0:
        # Just need to plus a number to the whole column
        return data[col] + 1
    else:
        # min < 1
        # Add the column with -min + 1
        return data[col] + (-min + 1)
