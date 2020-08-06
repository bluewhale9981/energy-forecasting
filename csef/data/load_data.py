from pathlib import Path

import pandas as pd
import numpy as np


def load_data(data_path=None):

    if not data_path:
        data_path = Path('..', '..', 'data', 'raw')

    consumption_train = pd.read_csv(data_path / 'consumption_train.csv',
                                    index_col=0, parse_dates=['timestamp'])
    cold_start_test = pd.read_csv(data_path / 'cold_start_test.csv',
                                  index_col=0, parse_dates=['timestamp'])
    submission_format = pd.read_csv(data_path / 'submission_format.csv',
                                    index_col='pred_id',
                                    parse_dates=['timestamp'])
    meta = pd.read_csv(data_path / 'meta.csv', index_col=0)

    return {
        'consumption_train': consumption_train,
        'cold_start_test': cold_start_test,
        'submission_format': submission_format,
        'meta': meta
    }


def sampling_data(df, frac=0.01, RANDOM_SEED=2018):
    rng = np.random.RandomState(seed=RANDOM_SEED)
    series_ids = df.series_id.unique()
    series_mask = rng.binomial(1, frac, size=series_ids.shape).astype(bool)

    training_series = series_ids[series_mask]

    # reduce training data to series subset
    return df[df.series_id.isin(training_series)]


def train_test_split(df, n_test=24, group_col='series_id'):
    df = df.copy()

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for ser_id, ser_data in df.groupby(group_col):
        split_train, split_test = ser_data[:-n_test], ser_data[-n_test:]

        split_train[group_col] = ser_id
        split_test[group_col] = ser_id

        train_df = train_df.append(split_train)
        test_df = test_df.append(split_test)

    return train_df, test_df


def describe_training_data(train_df):
    num_training_series = train_df.series_id.nunique()
    num_training_days = num_training_series * 28
    num_training_hours = num_training_days * 24

    desc = f'There are {num_training_series} training ' \
           f'series totaling {num_training_days} days ' \
           f'({num_training_hours} hours) of consumption data.'

    print(desc)


def save_submission(submission_df, file_path):
    submission_df.to_csv(file_path, index_label='pred_id')
