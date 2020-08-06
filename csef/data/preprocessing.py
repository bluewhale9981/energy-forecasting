# math and data manipulation
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_lagged_features(df, lag=1, train_col='consumption'):
    if not type(df) == pd.DataFrame:
        df = pd.DataFrame(df, columns=[train_col])

    def _rename_lag(ser, j):
        ser.name = ser.name + f'_{j}'
        return ser

    # add a column lagged by `i` steps
    for i in range(1, lag + 1):
        df = df.join(df['consumption'].shift(i).pipe(_rename_lag, i))

    df.dropna(inplace=True)
    return df


def prepare_training_data(df, lag, train_col='consumption'):
    """ Converts a series of consumption data into a
        lagged, scaled sample.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_vals = scaler.fit_transform(df.values.reshape(-1, 1))

    # convert consumption series to lagged features
    df_lagged = create_lagged_features(df_vals, lag=lag, train_col=train_col)

    # X, y format taking the first column (original time series) to be the y
    X = df_lagged.drop(train_col, axis=1).values
    y = df_lagged[train_col].values

    # keras expects 3 dimensional X
    X = X.reshape(X.shape[0], 1, X.shape[1])

    return X, y, scaler
