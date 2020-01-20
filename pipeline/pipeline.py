import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class Pipeline:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _calculate_score(predictions: np.ndarray, correct: np.ndarray) -> float:
        """Calculates the Root Mean Squared Logarithmic Error (RMSLE)

        :param predictions: array of shape (X,) containing the predictions
        :param correct: array of shape (X,) containing the true labels
        :return: float value representing the score
        """
        return np.sqrt(np.sum(np.square(np.log(predictions + 1) - np.log(correct + 1))) / len(correct))

    def run(self, df: pd.DataFrame, model: Any):
        """Runs the model on the given dataframe and logs the results

        :param df: dataframe
        :param model: model
        :return:
        """
        print(f'Running cross validation with the following model:\n{model}')

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        date_1 = datetime.datetime(year=2016, month=1, day=1)
        date_2 = datetime.datetime(year=2016, month=4, day=1)
        date_3 = datetime.datetime(year=2016, month=7, day=1)
        date_4 = datetime.datetime(year=2016, month=10, day=1)
        date_5 = datetime.datetime(year=2017, month=1, day=1)

        for train_start, train_end, test_start, test_end in [(date_1, date_2, date_2, date_3),
                                                             (date_1, date_3, date_3, date_4),
                                                             (date_1, date_4, date_4, date_5)]:
            print('Calculating train and test datasets')
            train_df = df[(df['timestamp'] >= train_start) & (df['timestamp'] < train_end)]
            test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)]

            columns = list(train_df.columns)
            columns.remove('timestamp')
            columns.remove('meter_reading')

            print(columns)

            train_data = train_df[columns]
            test_data = test_df[columns]

            print(f'Fitting the model on train dataset of size {len(train_data)}')
            model.fit(train_data, train_df['meter_reading'])
            print(f'Predicting for test dataset of size {len(test_data)}')
            predictions = model.predict(test_data)

            score = self._calculate_score(predictions, test_df['meter_reading'])
            print(f'Score: {score}')


# TODO: switch to correct score function
# TODO: output logs to log files


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run(df=pd.read_csv('../resources/train.csv'),
                 model=RandomForestRegressor(n_estimators=4))
