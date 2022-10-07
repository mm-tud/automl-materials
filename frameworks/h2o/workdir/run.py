#!/usr/bin/env python
# coding: utf-8

from h2o.automl import H2OAutoML
import h2o
from sklearn.model_selection import ShuffleSplit
from datetime import datetime
from contextlib import redirect_stdout
import sys
import sklearn.metrics as metrics
import pickle
import pandas as pd
import os
import numpy as np
import csv
INNER_SPLITS = 10
DATA_FOLDER = '/data'
DEFAULT_TRAIN_SIZE = 0.75
NAME_DATA = 'data.csv'
NAME_TEST = 'test.csv'
NAME_TRAIN = 'train.csv'
OUTER_SPLITS = 5
NUM_CORES = 8
MAX_TIME_MINUTES = 60
SEED = 1


def regression_results(y_true, y_pred):
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MAPE: ', round(mape, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))

    return([r2, np.sqrt(mse), mean_absolute_error, mse, mape])


def check_dataset(path):
    try:
        dirs = [
            d for d in os.listdir(path) if os.path.isdir(
                os.path.join(
                    path, d))]
        files = [
            f for f in os.listdir(path) if os.path.isfile(
                os.path.join(
                    path, f))]
    except NotADirectoryError:
        print(path, 'is not a dataset directory!')
        dirs = []
        files = []

    if NAME_TEST in files:
        data_type = 'split'
    elif NAME_DATA in files:
        data_type = 'full'
    else:
        data_type = None

    return dirs, files, data_type


def get_X_and_y_and_train_size(path):
    try:
        with open(os.path.join(path, 'X.txt'), 'r') as f:
            X = f.read().rstrip('\n').split(',')
    except FileNotFoundError:
        print('X.txt not found!')
        X = None

    try:
        with open(os.path.join(path, 'y.txt'), 'r') as f:
            y = f.read().rstrip('\n').split(',')
    except FileNotFoundError:
        print('y.txt not found!')
        y = None

    try:
        with open(os.path.join(path, 'train_size.txt'), 'r') as f:
            train_size = float(f.read().rstrip('\n'))
            print('train_size for outer loop = {}'.format(train_size))
    except FileNotFoundError:
        print('train_size.txt not found! train_size is set to {}'.format(
            DEFAULT_TRAIN_SIZE))
        train_size = DEFAULT_TRAIN_SIZE
    return X, y, train_size


def read_data(name, path, data_type):

    def fix_categorical(df):
        # prevent unknown input type error for string input types
        stringcols = df.select_dtypes(include='object').columns
        df[stringcols] = df[stringcols].astype('category')
        return df

    X_column, y_column, train_size = get_X_and_y_and_train_size(
        os.path.join(path, name))
    if data_type == 'split':
        df_train = pd.read_csv(os.path.join(path, NAME_TRAIN),
                               delimiter=';')
        df_test = pd.read_csv(os.path.join(path, NAME_TEST),
                              delimiter=';')
        df_train = fix_categorical(df_train)
        df_test = fix_categorical(df_test)
        X = df_train[X_column]
        X_test = df_test[X_column]
        y = df_train[y_column]
        y_test = df_test[y_column]
        return X, X_test, y, y_test, train_size
    elif data_type == 'full':
        df = pd.read_csv(os.path.join(path, NAME_DATA),
                         delimiter=';')
        df = df.sample(frac=1).reset_index(drop=True)
        df = fix_categorical(df)
        X = df[X_column]
        y = df[y_column]
        return X, None, y, None, train_size
    else:
        raise FileNotFoundError


def create_output_dir(name):
    time_run = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    folder_name = '{name}_{time}'.format(name=name, time=time_run)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def framework_train(output_dir, X, y):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    h2o.init()
    # h2o needs to read data in its own format, so
    # 1. import complete dataset into pandas
    # 2. train test split
    # 3. save splits into csv
    # 4. import split csvs in h2o
    #X_column, y_column, _ = get_X_and_y_and_train_size(os.path.join(path, name))
    X_column = X.columns
    y_column = y.columns

    data = X.merge(y,
                   left_index=True,
                   right_index=True,
                   how='outer')

    data.to_csv(os.path.join(output_dir, NAME_TRAIN))
    h2o_data = h2o.import_file(os.path.join(output_dir, NAME_TRAIN))
    os.remove(os.path.join(output_dir, NAME_TRAIN))
    h2o_columns = h2o_data.columns
    h2o_columns.remove(y_column[0])

    automl = H2OAutoML(  # max_models=20,(only time restriction
        max_runtime_secs=MAX_TIME_MINUTES * 60,
        nfolds=INNER_SPLITS,
        seed=SEED)

    print(datetime.now())
    print('Training over {} min started'.format(MAX_TIME_MINUTES))
    automl.train(x=h2o_columns,
                 y=y_column[0],
                 training_frame=h2o_data)
    print(datetime.now())

    lb = automl.leaderboard
    print(lb.head(rows=lb.nrows))  # print all, not only default (10)

    best_model = automl.get_best_model()
    print(best_model)
    with open(os.path.join(output_dir, 'best_model.txt'), 'w') as f:
        with redirect_stdout(f):
            print(best_model)

    with open(os.path.join(output_dir, 'automl.pickle'), 'wb') as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    return automl


def main(name, path, data_type):
    print(name)
    try:
        X, X_test, y, y_test, train_size = read_data(name, path, data_type)
    except FileNotFoundError:
        return
    output_dir = create_output_dir(name)

    scores = [['r2', 'rmse', 'mae', 'mse', 'mape']]
    if data_type == 'full':
        # "outer loop CV"
        rs = ShuffleSplit(
            n_splits=OUTER_SPLITS,
            train_size=train_size,
            random_state=SEED)
        rs.get_n_splits(X)
        i = 1
        for train_index, test_index in rs.split(X):
            print(10 * '-', 'SPLIT {}'.format(i), 10 * '-')
            subpath = os.path.join(output_dir, 'split_{}'.format(i))
            automl = framework_train(
                subpath, X.loc[train_index], y.loc[train_index])

            # --- FRAMEWORK SPECIFIC START ------------------------------------
            # format test data to h2o format
            test_data = X.loc[test_index].merge(y.loc[test_index],
                                                left_index=True,
                                                right_index=True,
                                                how='outer')
            test_data.to_csv(os.path.join(output_dir, NAME_TEST))
            h2o_test_data = h2o.import_file(
                os.path.join(output_dir, NAME_TEST))
            y_pred = automl.predict(h2o_test_data).as_data_frame()
            # --- FRAMEWORK SPECIFIC END --------------------------------------

            y_true = y.loc[test_index]
            scores.append(regression_results(y_true, y_pred))
            i += 1
            h2o.remove_all()
            print()
    else:
        automl = framework_train(output_dir, X, y)

        # --- FRAMEWORK SPECIFIC START ----------------------------------------
        # format test data to h2o format
        test_data = X_test.merge(y_test,
                                 left_index=True,
                                 right_index=True,
                                 how='outer')
        test_data.to_csv(os.path.join(output_dir, NAME_TEST))
        h2o_test_data = h2o.import_file(os.path.join(output_dir, NAME_TEST))
        y_pred = automl.predict(h2o_test_data).as_data_frame()
        # --- FRAMEWORK SPECIFIC END ------------------------------------------

        #y_pred = automl.predict(X_test)
        y_true = y_test
        scores.append(regression_results(y_true, y_pred))
        h2o.remove_all()

    with open(os.path.join(output_dir, 'regression_summary.csv'),
              'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(scores)


if __name__ == '__main__':
    for dataset in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, dataset)
        dirs, files, data_type = check_dataset(path)

        print(79 * 'v')
        print('Dataset name: {d}, type: {t}'.format(d=dataset,
                                                    t=data_type))
        print()

        for task in dirs:
            print(39 * '-')
            print('Task name: {}'.format(task))
            main(task, path, data_type)
        print(79 * '^')
