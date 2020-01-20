import datetime
import os
import time
from typing import Any
from typing import List
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class FoldSummary(NamedTuple):
    train_start: datetime.datetime
    train_end: datetime.datetime
    test_start: datetime.datetime
    test_end: datetime.datetime
    score: float


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

        summaries: List[FoldSummary] = []

        for train_start, train_end, test_start, test_end in [
            (date_1, date_2, date_2, date_3),
            # (date_1, date_3, date_3, date_4),
            # (date_1, date_4, date_4, date_5)
        ]:
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

            summaries.append(FoldSummary(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                score=score
            ))

        filename = f'../resources/runs/{time.time()}.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w+') as f:
            f.write(f'{model.__class__.__name__}\n')
            f.write(f'{str(model.get_params())}\n')
            for summary in summaries:
                f.write(f'Summary (\n'
                        f'\ttrain start: {summary.train_start}\n'
                        f'\ttrain end: {summary.train_end}\n'
                        f'\ttest start: {summary.test_start}\n'
                        f'\ttest end: {summary.test_end}\n'
                        f'\tscore: {summary.score}\n'
                        f')\n')

        print(summaries)

        return model

    def create_submission(self, df: pd.DataFrame, model: Any):
        columns = list(df.columns)
        columns.remove('timestamp')
        columns.remove('row_id')

        predictions = model.predict(df[columns])

        with open('submission.txt', 'w+') as f:
            f.write('row_id,meter_reading\n')
            for row_id, prediction in zip(df['row_id'], predictions):
                f.write(f'{row_id},{prediction}\n')


if __name__ == '__main__':
    pipeline = Pipeline()
    model = pipeline.run(df=pd.read_csv('../resources/train.csv'),
                         model=RandomForestRegressor(n_estimators=10))

    pipeline.create_submission(df=pd.read_csv('../resources/test.csv'), model=model)
