# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorflow import keras

from csef.utils.logging import getLogger
from csef.data import preprocessing


logger = getLogger(logger_name=__name__)


class BaseModel(object):

    default_config = {
        'loss': 'mean_absolute_error',
        'optimizer': 'adam',
        'stateful': True,
        'loss': 'mean_absolute_error',
        'train_col': 'consumption',
        'group_col': 'series_id'
    }

    def __init__(self, config, is_init_model=True):

        # Update the default config
        self.default_config.update(config)

        config = self.default_config

        self.config = config

        self.n_input = config['n_input']
        self.n_nodes = config['n_nodes']
        self.n_batch = config['n_batch']

        self.loss = config['loss']
        self.optimizer = config['optimizer']
        self.stateful = config['stateful']
        self.train_col = config['train_col']
        self.group_col = config['group_col']

        if is_init_model:
            self.model = self._build_model()

    def _build_model(self):
        raise NotImplemented('Need override this method!')

    def reset_memory(self):
        # TODO: implement
        raise NotImplemented("TODO")

    def reset_states(self):
        self.model.reset_states()

    def fit(self, train_df):
        n_series = train_df[self.group_col].nunique()

        for ser_id, ser_data in tqdm(train_df.groupby(self.group_col),
                                     total=n_series,
                                     desc="Fitting the data"):
            # prepare the data
            X, y, _ = preprocessing.prepare_training_data(ser_data[self.train_col], self.n_input)
            self.model.fit(X, y, epochs=1, batch_size=self.n_batch, verbose=0, shuffle=False)
            self.model.reset_states()

        return self

    def predict(self, df, scaler, num_pred_hours=24, is_inverse_transform=True):

        # allocate prediction frame
        preds_scaled = np.zeros(num_pred_hours)

        # initial X is last lag values from the cold start
        X = scaler.transform(df.values.reshape(-1, 1))[-self.n_input:]

        # forecast
        for i in range(num_pred_hours):
            # predict scaled value for next time step
            yhat = self.model.predict(X.reshape(1, 1, self.n_input), batch_size=1)[0][0]
            preds_scaled[i] = yhat

            # update X to be latest data plus prediction
            X = pd.Series(X.ravel()).shift(-1).fillna(yhat).values

        # revert scale back to original range
        if is_inverse_transform:
            hourly_preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        else:
            hourly_preds = preds_scaled

        return hourly_preds

    def make_submission(self, submission_df, cold_start_test):
        """
        make the submission file
        :param submission_df: The submission sample
        :param cold_start_test: The test data
        :return: The submission df
        """
        my_submission_df = submission_df.copy()

        pred_window_to_num_preds = {'hourly': 24, 'daily': 7, 'weekly': 2}
        pred_window_to_num_pred_hours = {'hourly': 24, 'daily': 7 * 24, 'weekly': 2 * 7 * 24}

        num_test_series = my_submission_df[self.group_col].nunique()

        self.model.reset_states()

        for ser_id, pred_df in tqdm(my_submission_df.groupby(self.group_col),
                                    total=num_test_series,
                                    desc="Forecasting from Cold Start Data"):
            # get info about this series' prediction window
            pred_window = pred_df.prediction_window.unique()[0]
            num_preds = pred_window_to_num_preds[pred_window]
            num_pred_hours = pred_window_to_num_pred_hours[pred_window]

            # prepare cold start data
            series_data = cold_start_test[cold_start_test[self.group_col] == ser_id][self.train_col]
            cold_X, cold_y, scaler = preprocessing.prepare_training_data(series_data, self.n_input)

            # fine tune our lstm model to this site using cold start data
            self.model.fit(cold_X, cold_y, epochs=1, batch_size=self.n_batch, verbose=0, shuffle=False)

            # make hourly forecasts for duration of pred window
            preds = self.predict(series_data, scaler, num_pred_hours=num_pred_hours)

            # reduce by taking sum over each sub window in pred window
            reduced_preds = [pred.sum() for pred in np.split(preds, num_preds)]

            # store result in submission DataFrame
            ser_id_mask = my_submission_df[self.group_col] == ser_id
            my_submission_df.loc[ser_id_mask, self.train_col] = reduced_preds

        return my_submission_df

    def save_model(self, model_path):
        keras.models.save_model(
            self.model,
            model_path
        )

    def load_model(self, model_path):
        self.model = keras.models.load_model(
            model_path
        )


class GeneralModel(BaseModel):
    """This get the config to build model"""

    def __init__(self, config):
        assert 'model' in config

        # Each model definition including
        # layer_type: Dense, LSTM ...
        # layer_config: {}
        self.model_definitions = config['model']

        super().__init__(config)

    def __layer_type_mapping(self, layer_type):
        return getattr(keras.layers, layer_type)

    def _build_model(self):

        # instantiate a sequential model
        model = keras.Sequential()

        for model_definition in self.model_definitions:
            layer_type = model_definition['layer_type']
            layer_config = model_definition['layer_config']

            layer_type_class = self.__layer_type_mapping(layer_type)

            model.add(layer_type_class(**layer_config))

        # compile
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model


