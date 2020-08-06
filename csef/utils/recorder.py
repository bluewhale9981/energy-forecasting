# -*- coding: utf-8 -*-

from __future__ import absolute_import
from csef.utils.aws_storage import S3Storage
from csef.utils.logging import getLogger


class PipelineRecorder(object):
    """docstring for PipelineRecorder"""
    def __init__(self, version, timestamp):
        self.version = version
        self.timestamp = timestamp

        self.path = 'history/pipeline_{0}_{1}'.format(version, timestamp)
        self.logger = getLogger(logger_name=__name__)

    def write(self, line):
        """
        Write a new line to history.
        """
        with open(self.path, 'a') as the_file:
            the_file.write('%s\n' % line)

    def upload(self):
        """
        Upload pipeline to AWS S3
        """
        remote_path = self.path
        s3_storage = S3Storage()

        if s3_storage is not None:
            self.logger.info('Upload local %s history to S3 remote %s' %(self.path, remote_path))
            s3_storage.upload(self.path, remote_path)


class VersionRecorder(object):
    """docstring for VersionRecorder"""
    def __init__(self, version):
        self.version = version
        self.path = 'csef/features/%s/history' % self.version

    def write(self, line):
        """
        Write a new line to history.
        """
        with open(self.path, 'a') as the_file:
            the_file.write('%s\n' % line)
