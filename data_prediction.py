import pandas as pd
import data_preparation as dp
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from  sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

pd.set_option('display.max_columns', 100)


def get_data() -> pd.DataFrame:
    return dp.prepare_data()


def add_polynomials(X: pd.DataFrame) -> pd.DataFrame:
    columns = dict(zip(range(0, len(X.columns)), list(X.columns)))
    for col_1 in list(columns.keys()):
        for col_2 in list(columns.keys())[col_1:]:
            X[columns.get(col_1) + ' and ' + columns.get(col_2)] = \
                X[columns.get(col_1)] * X[columns.get(col_2)]

    return X


def X_y_split(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

    X = data.drop('Rating', axis=1).reset_index(drop=True)
    X = add_polynomials(X)

    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    y = data['Rating']

    return X, y


def train(X_train: pd.DataFrame, y_train: pd.Series) -> RidgeCV:

    lm = RidgeCV(alphas=np.linspace(0.01, 10, 24))
    lm.fit(X_train, y_train)
    return lm


def predict():
    X, y = X_y_split(get_data())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = train(X_train, y_train)
    predictions = model.predict(X_test)
    print(round((len(predictions[predictions < 3.5]) / len(predictions)), 2))
    print('Mean of test set: {}'.format(round(mean_absolute_error(y_test, predictions), 2)))
    my_preds = list()
    for i in range(0, len(X_test)):
        my_preds.append(round(abs(model.predict(X_test[i].reshape(1, -1))[0] - y_test.iloc[i]), 2))
    print('Median of X_test: {}'.format(np.median(my_preds)))


if __name__ == '__main__':
    predict()
