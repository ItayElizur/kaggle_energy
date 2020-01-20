from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


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
        return np.sqrt(np.sum(np.square(np.log(predictions + 1) - np.log(targets[correct] + 1))) / len(correct))

    def run(self, data: np.ndarray, targets: np.ndarray, model: Any, folds=5):
        print(f'Running cross validation with the following model:\n{model}')

        kf = KFold(folds, shuffle=True)
        print(f'Starting K fold cross validation with {folds} folds')
        for idx, (train, test) in enumerate(kf.split(data, targets)):
            print(f'Starting fold number {idx + 1}')
            model.fit(data[train], targets[train])
            predicted = model.predict(data[test])
            score = self._calculate_score(predicted, targets[test])
            print(score)


# TODO: switch to correct score function
# TODO: output logs to log files


if __name__ == '__main__':
    pipeline = Pipeline()
    data = np.random.rand(10000, 5)
    targets = np.random.rand(10000) * 250

    print(Pipeline()._calculate_score(np.array([100] * 100), np.array([200] * 100)))

    pipeline.run(data=data,
                 targets=targets,
                 model=RandomForestRegressor(n_estimators=20))
