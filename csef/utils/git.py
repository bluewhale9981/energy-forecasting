# -*- coding: utf-8 -*-
import subprocess
import os


def get_global_username():
    """Get global username"""
    try:
        return subprocess.\
            check_output(['git', 'config', 'user.name']).\
            decode("utf-8").\
            replace('\n', '')
    except:
        return os.environ.get('JOB_OWNER', "")


def get_commit_id():
    """Get the current commit id"""
    try:
        return subprocess.\
            check_output(['git', 'rev-parse', '--short', 'HEAD']).\
            decode("utf-8").\
            replace('\n', '')
    except:
        return ""
