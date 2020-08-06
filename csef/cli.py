# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
from dotenv import load_dotenv

import click
from csef.session import SessionManager
from csef.pipeline import PipelineStorageManager, PipelineManager
from csef.utils.helper import load_config

from csef.data.preprocessing import preprocess_raw_data


load_dotenv('.env')


@click.group()
@click.option(
    '-opt/-no-opt', '--option/--no-option',
    help='Run command with option or not',
    default=True
)
@click.pass_context
def pip_cli(ctx, **kwargs):
    ctx.obj = kwargs
    SessionManager().renew(kwargs)


@pip_cli.command(name='do_something')
def do_something():

    print('do something')


@pip_cli.command(name='preprocessing')
def preprocessing():
    print('perform preprocessing pipeline')
    preprocess_raw_data()


@pip_cli.command(name='upload-data')
@click.option(
    '-v', '--data-version',
    prompt='Data version',
    required=True,
    help='The data transformed version. E.g: v1.'
)
@click.option(
    '-t', '--data-tag',
    default='stable',
    help='The tag of data.'
)
@click.option(
    '-compress/-no-compress',
    help='Flag indicates to compress data before upload or not',
    default=True
)
@click.pass_obj
def pip_upload_data(ctx, data_version, data_tag, compress):
    PipelineStorageManager().upload_data_files(data_version, data_tag, compress)


@pip_cli.command(name='run')
@click.option(
    '-cf', '--config-file',
    prompt='Config file',
    required=True,
    help='The path to config file need to be processed'
)
@click.pass_obj
def pip_run(ctx, **kwargs):
    # Set the environment variable for the process running on the cloud
    proj_home = os.path.dirname(os.path.realpath(__file__))
    gcloud_creds_path = os.path.join(proj_home, '../gcloud-creds.json')

    os.environ.setdefault('PROJ_HOME', proj_home)
    os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', gcloud_creds_path)

    if 'owner' in kwargs:
        os.environ.setdefault('JOB_OWNER', kwargs['owner'])

    # Merge 2 params
    ctx.update(kwargs)
    args = ctx

    # Init the variables
    config_file = args['config_file']

    configs = load_config(config_file)

    # Start the pipeline
    PipelineManager() \
        .init(configs, args) \
        .run() \
        .finish()


def main():
    pip_cli()
