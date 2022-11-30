from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.timeseries.splitter import MultiWindowSplitter
import pandas as pd


def load_daily_price_adjusted(filename: str) -> pd.DataFrame:
    """
    Loads from a file, and returns a Pandas dataframe, with the daily information for a given stock, including its
    adjusted prices.
    :param filename: the name of the file with the information.
    :return: the requested dataset in a dataframe.
    """
    df = pd.read_csv(filename)
    df = df.astype({'timestamp': 'datetime64'})
    return df


def main():
    df = load_daily_price_adjusted('daily_price-QQQ.csv')
    df.drop(['open', 'high', 'low', 'close', 'volume', 'dividend_amount', 'split_coefficient'], axis=1,
            inplace=True)
    df['symbol'] = 'QQQ'
    ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='symbol', timestamp_column='timestamp')

    prediction_length = 5
    num_windows = 52
    train_data = ts_df.slice_by_timestep(None, -prediction_length * num_windows)
    autogluon_dir = 'autogluon'

    splitter = MultiWindowSplitter(num_windows=num_windows)
    predictor = TimeSeriesPredictor(path=autogluon_dir,
                                    target="adjusted_close",
                                    prediction_length=prediction_length,
                                    eval_metric='MAPE',
                                    ignore_time_index=True,
                                    validation_splitter=splitter
                                    )

    predictor.fit(train_data,
                  presets='fast_training',
                  time_limit=90
                  )

    leaderboard = predictor.leaderboard(silent=False)
    summary = predictor.fit_summary()  # raises KeyError: 'score'
    info = predictor.info() # AttributeError: 'str' object has no attribute 'name'


if __name__ == '__main__':
    main()

""" TODO Open questions
When using multi-window backtesting, is the validation metric the average of the validation metrics?
When using multi-window backtesting, is the model re-trained on all the available data?
"""
