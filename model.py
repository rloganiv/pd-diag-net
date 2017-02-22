"""Trains an LSTM model on the PaHaW dataset and evaluates using cross validation"""
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

import numpy as np
import process
import pdb


class Splits(object):
    """ Stores the kfold splits

    Attributes:
        train: list. A list of the training splits
        test: list. A list of the testing splits

    """
    train = []
    test = []


def transform_position(data):
    """Transform position values into diplacement values.

    Args:
        data: dictionary. Contains our data where the key is the subject. See process.py for more information.
    """
    for subject in data.itervalues():
        for task_data in subject.task.itervalues():
            # Subtract the previous time point's coordinates
            task_data[1:, 0:2] -= task_data[0:-1, 0:2]
            # Set starting coordinates to (0, 0)
            task_data[0,0:2] = 0


def pad_sequence(data, max_len=500):
    """Pad data so all sequences have uniform length. This is done by adding sequences with zeros. We also
        add an extra binary feature that is 1 when data is availble at that
        time and 0 when we are padding.
        If a time sequence is over max_len, the extra time points are removed.

    Args:
        data: dictionary. Contains our data where the key is the subject. See process.py for more information.
        max_len: int. Max length of all time series.
    """
    for subject in data.itervalues():
        for task_key, task_data in subject.task.iteritems():
            if task_data.shape[0] < max_len:
                task_data = np.hstack((task_data, np.ones(shape=(task_data.shape[0], 1))))
                task_data = np.vstack((task_data, np.zeros(shape=(max_len - task_data.shape[0], 8))))
            else:
                task_data = task_data[0:max_len,:]
                task_data = np.hstack((task_data, np.ones(shape=(task_data.shape[0], 1))))
            subject.task[task_key] = task_data


def remove_time(data):
    """ Removes the time feature from our data since this is the same across the data

    Args:
        data: dictionary. Contains our data where the key is the subject. See process.py for more information.
    """
    for subject in data.itervalues():
        for task_key, task_data in subject.task.iteritems():
            task_data = np.delete(task_data, 2, axis=1)
            subject.task[task_key] = task_data


def feature_extraction(data):
    """Generates features from raw data. This leaves most of the data the same but involves several changes.
        - Transforming Cartesian coordinate features into delta features (change in position).
        - Pads sequences to have the same length.
        - Removes the time feature.
        See functions for more detail.

    Args:
        data: dictionary. Contains our data where the key is the subject. See process.py for more information.
    """
    transform_position(data)
    pad_sequence(data, max_len=16071)   # maximum sequence length of data is 16071, hardcoded
    remove_time(data)


def KFold(data, n_splits=10):
    """Performs a K-Fold split of our data

    Args:
        data: dictionary. Contains our data where the key is the subject. See process.py for more information.

    Returns:
        kf: list. Contains a dictionary for each split. Each dictionary has a
        'train' and 'test' key indicating the
            subjects of each set.
    """
    from copy import copy
    from math import ceil
    import random

    subjs = {}
    for key in data:
        subjs[key] = data[key].info['PD status']
    remaining = copy(subjs)

    kf = Splits()
    for i in xrange(n_splits):
        train = copy(subjs)
        if i == n_splits-1:
            test = list(remaining.keys())
        else:
            test = random.sample(remaining, len(data)/n_splits)
        for key in test:
            remaining.pop(key, None)
            train.pop(key, None)

        kf.test.append(test)
        kf.train.append(list(train.keys()))

    return kf


def create_datasets(data, train, test):
    """Creates the training and testing datasets from the subjects consitituting each set.

    Args:
        data: dictionary. Contains our data where the key is the subject. See process.py for more information.
        train: list. A list of subjects in the training set
        test: list. A list of subjects in the testing set. Subjects in the testing set are mutually exclusive from the
            training set.
    Returns:
        X_train, y_train, X_test, y_test: numpy arrays. Features and predictions for the training and test set.
    """
    num_train = 0
    num_test = 0

    # Get the number of training and testing sequences
    for subject_id in train:
        num_train += len(data[subject_id].task)
    for subject_id in test:
        num_test += len(data[subject_id].task)

    # Hard-coded train and test sets
    X_train = np.empty(shape=(num_train, 16071 ,7))
    y_train = np.empty(shape=(num_train, 1))
    X_test = np.empty(shape=(num_test, 16071, 7))
    y_test = np.empty(shape=(num_test, 1))

    # Extract training and test sets from data dictionary
    for idx, subject_id in enumerate(train):
        for task_data in data[subject_id].task.itervalues():
            task_data = task_data.reshape((1, task_data.shape[0], task_data.shape[1]))
            X_train[idx] = task_data
            y_train[idx] = int(data[subject_id].info['PD status'])

    for idx, subject_id in enumerate(test):
        for task_data in data[subject_id].task.itervalues():
            task_data = task_data.reshape((1, task_data.shape[0], task_data.shape[1]))
            X_test[idx] = task_data
            y_test[idx] = int(data[subject_id].info['PD status'])

    return X_train, y_train, X_test, y_test


def normalize_data(X_train, X_test):
    """ Normalizes features in the training set to be in range [0, 1]. Then applies transformation
        to test set.

    Args:
        X_train: numpy array. 3D matrix containing our training sequences
        X_test: numpy array. 3D array containing our test sequences
    """
    max_feature_val = np.max(np.max(X_train, axis=1), axis=0) # highest feature values in training set
    X_train /= max_feature_val
    X_test /= max_feature_val

def evaluate_model():
    """ Loads the data dictionary and then performs K-Fold cross validation to obtain the classification accuracy
        of our model.
    """
    data = process.load_dataset()
    feature_extraction(data)

    kf_splits = KFold(data)
    sum_scores = 0
    for i in xrange(len(kf_splits.train)):
        print "========= Cross Validation Iteration ", i +1, " out of ", len(kf_splits.train), " ========="
        # Data Processing and Normalization
        train = kf_splits.train[i]
        test = kf_splits.test[i]
        (X_train, y_train, X_test, y_test) = create_datasets(data, train, test)
        normalize_data(X_train, X_test)

        # Model Training
        model = Sequential()
        model.add(LSTM(100, input_dim=7))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  nb_epoch=3, batch_size=5)

        # Evaluation of model
        score = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        sum_scores += scores[1]
    print "========================="
    accuracy = sum_scores/len(kf_splits.train)
    print "Cross validation accuracy: ", accuracy

if __name__== '__main__':
    evaluate_model()
