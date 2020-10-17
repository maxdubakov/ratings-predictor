import pandas as pd
import data_preparation as dp
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from  sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.max_columns', 100)


def get_data() -> pd.DataFrame:
    return dp.prepare_data()


def train_cv_test_split(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

    data_len = len(data)
    train_size = int(round(0.6 * data_len))
    cv_size = int(round(0.2 * data_len))
    test_size = int(round(0.2 * data_len))

    X = data.drop('Rating', axis=1).reset_index(drop=True)
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    y = data['Rating']

    return (X[:train_size], y[:train_size],
            X[train_size:train_size + cv_size], y[train_size:train_size + cv_size],
            X[train_size + cv_size:train_size + cv_size + test_size],
            y[train_size + cv_size:train_size + cv_size + test_size])


def train(X_train: pd.DataFrame, y_train: pd.Series) -> RidgeCV:

    lm = RidgeCV(alphas=np.linspace(0.01, 10, 24))
    lm.fit(X_train, y_train)
    return lm


def predict():
    X_train, y_train, X_cv, y_cv, X_test, y_test = train_cv_test_split(get_data())
    model = train(X_train, y_train)
    predictions = model.predict(X_test)
    print(mean_squared_error(y_test, predictions))
    for i in range(0, 100):
        print(model.predict(X_cv[i].reshape(1, -1)), y_cv.iloc[i])


if __name__ == '__main__':
    predict()
