#!/usr/bin/env python
# coding: utf-8

from autosklearn.regression import AutoSklearnRegressor as Regressor
from autosklearn.metrics import median_absolute_error as mabse
from autosklearn.metrics import mean_absolute_error as mae
from autosklearn.metrics import mean_squared_error as mse
from autosklearn.metrics import r2
import autosklearn.regression
from sklearn.model_selection import ShuffleSplit
from datetime import datetime
from contextlib import redirect_stdout
import sys
import sklearn.metrics as metrics
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import gc
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


# --- FRAMEWORK SPECIFIC START --- #0 -------------------------------------
def get_metric_result(cv_results):
    results = pd.DataFrame.from_dict(cv_results)
    results = results[results['status'] == "Success"]
    cols = ['rank_test_scores', 'param_regressor:__choice__',
            'mean_test_score']
    cols.extend([key for key in cv_results.keys()
                if key.startswith('metric_')])
    return results[cols]
# --- FRAMEWORK SPECIFIC END ----- #0 -------------------------------------


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
    # --- FRAMEWORK SPECIFIC START --- #1 -------------------------------------
    auto_tmp_folder = os.path.join(output_dir, 'autosklearn_tmp')

    automl = Regressor(time_left_for_this_task=MAX_TIME_MINUTES * 60,
                       per_run_time_limit=MAX_TIME_MINUTES * 5,
                       resampling_strategy='cv',
                       resampling_strategy_arguments={'folds': INNER_SPLITS},
                       n_jobs=NUM_CORES,
                       seed=SEED,
                       scoring_functions=[r2, mse, mae, mabse],
                       tmp_folder=auto_tmp_folder,
                       delete_tmp_folder_after_terminate=True)
    # --- FRAMEWORK SPECIFIC END ----- #1 -------------------------------------

    print(datetime.now())
    print('Training over {} min started'.format(MAX_TIME_MINUTES))
    automl.fit(X, y)
    print(datetime.now())

    # --- FRAMEWORK SPECIFIC START --- #2 -------------------------------------
    print(automl.sprint_statistics())

    # plot train hist
    ra = np.arange(len(automl.cv_results_['status']))
    test_score = automl.cv_results_['mean_test_score']
    test_score[test_score < 0] = 0

    best = []
    for i in test_score:
        try:
            best.append(max(0, max(best), i))
        except ValueError:  # best is empty
            best.append(0)
    best = np.array(best)

    labels = []
    for i in automl.cv_results_['params']:
        labels.append(i['regressor:__choice__'])
    labels = np.array(labels)

    df = pd.DataFrame(dict(x=ra, y=test_score, label=labels))
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(15, 6))
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=6,
                label=name)
        ax.legend(frameon=True, framealpha=0.9)
        ax.plot(ra, best, color='gray')
        ax.set_title('Algorithm Performance')
        ax.set_ylabel('Score')
        ax.set_xlabel('Model No.')
    plt.savefig(os.path.join(output_dir, 'train_history.svg'))

   # save pkl, consumes to much storage
   # with open(os.path.join(output_dir, 'clf.pickle'), 'wb') as f:
   #     pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dir, 'stats.txt'), 'w') as f:
        with redirect_stdout(f):
            print(automl.sprint_statistics())

   # with open(os.path.join(output_dir, 'leaderboard.txt'), 'w') as f:
   #     lb = automl.leaderboard()
   #     with redirect_stdout(f):
   #         print(lb.to_string())

    with open(os.path.join(output_dir, 'metric_results.txt'), 'w') as f:
        with redirect_stdout(f):
            print(get_metric_result(automl.cv_results_).to_string(index=False))
    # --- FRAMEWORK SPECIFIC END ----- #2 -------------------------------------

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
            y_pred = automl.predict(X.loc[test_index])
            y_true = y.loc[test_index]
            scores.append(regression_results(y_true, y_pred))
            i += 1
            del automl
            gc.collect()
            print()
    else:
        automl = framework_train(output_dir, X, y)
        y_pred = automl.predict(X_test)
        y_true = y_test
        scores.append(regression_results(y_true, y_pred))
        del automl
        gc.collect()

    with open(os.path.join(output_dir, 'regression_summary.csv'),
              'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(scores)
    del X, X_test, y, y_test
    gc.collect()


if __name__ == '__main__':
    for dataset in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, dataset)
        dirs, files, data_type = check_dataset(path)

        print(79 * 'v')
        print('Dataset name: {d}, type: {t}'.format(d=dataset,
                                                    t=data_type))
        print()
        errorcount = 0
        done_tasks = ['_'.join(x.split('_', 2)[:2]) for x in os.listdir()]
        for task in dirs:
            print(39 * '-')
            print('Task name: {}'.format(task))
            if task not in done_tasks:
                try:
                    main(task, path, data_type)
                except BaseException:
                    print("error handling dataset")
                    errorcount += 1
                gc.collect()
                os.execl(sys.executable, 'python3', __file__, *sys.argv[1:])
    print("Finished with {} errors".format(errorcount))
    print(79 * '^')
