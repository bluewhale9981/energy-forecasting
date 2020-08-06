from tensorflow import keras
from .base import BaseModel


class SimpleLSTM(BaseModel):

    def _build_model(self):

        # model parameters
        batch_input_shape = (self.n_batch, 1, self.n_input)

        # instantiate a sequential model
        model = keras.Sequential()

        # add LSTM layer - stateful MUST be true here in
        # order to learn the patterns within a series
        model.add(keras.layers.LSTM(units=self.n_nodes,
                                    batch_input_shape=batch_input_shape,
                                    stateful=self.stateful))

        # followed by a dense layer with a single output for regression
        model.add(keras.layers.Dense(1))

        # compile
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model


class SummaryNodeLSTM(BaseModel):

    def _build_model(self):

        # model parameters
        batch_input_shape = (self.n_batch, 1, self.n_input)

        # instantiate a sequential model
        model = keras.Sequential()

        # add LSTM layer - stateful MUST be true here in
        # order to learn the patterns within a series
        model.add(keras.layers.LSTM(units=self.n_nodes,
                                    batch_input_shape=batch_input_shape,
                                    stateful=self.stateful))

        # second layer
        model.add(keras.layers.Dense(self.n_nodes, activation='relu'))

        # followed by a dense layer with a single output for regression
        model.add(keras.layers.Dense(1))

        # compile
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model


class StackedLSTM(BaseModel):

    def _build_model(self):

        # model parameters
        batch_input_shape = (self.n_batch, 1, self.n_input)

        # instantiate a sequential model
        model = keras.Sequential()

        # add LSTM layer - stateful MUST be true here in
        # order to learn the patterns within a series
        model.add(keras.layers.LSTM(units=self.n_nodes,
                                    batch_input_shape=batch_input_shape,
                                    activation='relu', return_sequences=True,
                                    stateful=self.stateful))

        # second layer
        model.add(keras.layers.LSTM(self.n_nodes, activation='relu', stateful=self.stateful))

        # followed by a dense layer with a single output for regression
        model.add(keras.layers.Dense(1))

        # compile
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model
