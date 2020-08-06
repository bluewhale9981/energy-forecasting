# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os
from boto3.session import Session


class S3Storage(object):
    def __init__(self):
        """
        Initial of S3Storage
        """
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION')
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        self.is_disabled = False

        if aws_key is None or \
                aws_secret is None or \
                aws_region is None or \
                bucket_name is None:
            print('There is no S3 config!')
            self.is_disabled = True
            return

        session = Session(
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region
        )
        self.s3_resource = session.resource('s3')
        self.s3_bucket = self.s3_resource.Bucket(bucket_name)
        self.bucket_name = bucket_name

    def upload(self, local_path, remote_path):
        """
        Upload a file from local to S3

        @Args:
        local_path (str): The local path of the file
        remote_path (str): The remote path of the file
        """
        if self.is_disabled is False:
            self.s3_bucket.upload_file(local_path, remote_path)

    def put_object(self, path, meta_json):
        """
        Put JSON object to S3
        """
        if self.is_disabled is False:
            obj = self.s3_resource.Object(self.bucket_name, path)
            obj.put(Body=meta_json)
