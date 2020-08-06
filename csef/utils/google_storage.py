# -*- coding: utf-8 -*-
from google.cloud import storage

from csef.utils.design_patterns import SingletonDecorator


@SingletonDecorator
class GoogleStorage(object):

    def __init__(self):
        """
        This class defines the wrapper for google datastore.
        """
        self.client = storage.Client()

    def create_bucket(self, bucket_name):
        """Creates a new bucket."""
        bucket = self.client.create_bucket(bucket_name)
        print('Bucket {} created'.format(bucket.name))

    def upload_blob(self, bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        storage_client = self.client
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print('File {} uploaded to {}.'.format(source_file_name, destination_blob_name))

    def download_blob(self, bucket_name, source_blob_name, destination_file_name):
        """Downloads a blob from the bucket."""
        storage_client = self.client
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)

        print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))

    def list_blobs_with_prefix(self, bucket_name, prefix, delimiter=None):
        """Lists all the blobs in the bucket that begin with the prefix.

        This can be used to list all blobs in a "folder", e.g. "public/".

        The delimiter argument can be used to restrict the results to only the
        "files" in the given "folder". Without the delimiter, the entire tree under
        the prefix is returned. For example, given these blobs:

            /a/1.txt
            /a/b/2.txt

        If you just specify prefix = '/a', you'll get back:

            /a/1.txt
            /a/b/2.txt

        However, if you specify prefix='/a' and delimiter='/', you'll get back:

            /a/1.txt

        """
        storage_client = self.client
        bucket = storage_client.get_bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

        return blobs
