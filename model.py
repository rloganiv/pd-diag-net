"""Trains an LSTM model on the PaHaW dataset and evaluates using cross validation"""

import numpy as np
import process

class Splits:
    """ Stores the kfold splits

    Attributes:
        train: list. A list of the training splits
        test: list. A list of the testing splits

    """
    train = []
    test = []
    
def transform_position(data):
    """Transforms the position values into delta values that indicate the change from the previous position
    
    Args:
        data: dictionary. Contains our data where the key is the subject. See process.py for more information.
    """
    for subject in data:
        for i in xrange(1, 9):
            task_id = 'task_%i' % i
            if hasattr(data[subject], task_id):
                x = getattr(data[subject], task_id)
                x[1:, 0:2] -= x[0:-1, 0:2]                  # Subtracts the previous time points coordinates
                x[0,0:2] = 0                                    # Sets starting coordinates to (0, 0)
                setattr(data[subject], task_id, x)

def pad_sequence(data, max_len=500):
    """Pad data so all sequences have uniform length. This is done by adding sequences with zeros. We also
        add an extra binary feature that is 1 when data is availble at that time and 0 when we are padding. 
        If a time sequence is over max_len, the extra time points are removed.

    Args:
        data: dictionary. Contains our data where the key is the subject. See process.py for more information.
        max_len: int. Max length of all time series. 
    """
    for subject in data:
        for i in xrange(1, 9):
            task_id = 'task_%i' % i
            if hasattr(data[subject], task_id):
                x = getattr(data[subject], task_id)

                if x.shape[0] < max_len:
                    x = np.hstack((x, np.ones(shape=(x.shape[0], 1))))                     # Append extra feature
                    x = np.vstack((x, np.zeros(shape=(max_len - x.shape[0], 8))))   # Adds extra time points, hardcoded
                    setattr(data[subject], task_id, x)
                else:
                    x = x[0:max_len,:]
                    x = np.hstack((x, np.ones(shape=(x.shape[0], 1))))
                    setattr(data[subject], task_id, x)

def remove_time(data):
    """ Removes the time feature from our data since this is the same across the data

    Args:
        data: dictionary. Contains our data where the key is the subject. See process.py for more information.
    """
    for subject in data:
        for i in xrange(1, 9):
            task_id = 'task_%i' % i
            if hasattr(data[subject], task_id):
                x = getattr(data[subject], task_id)
                x = np.delete(x, 2, axis=1)
                setattr(data[subject], task_id, x)

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
        kf: list. Contains a dictionary for each split. Each dictionary has a 'train' and 'test' key indicating the 
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
    for subj in train:
            for i in xrange(1, 9):
                if hasattr(data[subj], 'task_%i' % i):
                    num_train += 1
    for subj in test:
            for i in xrange(1, 9):
                if hasattr(data[subj], 'task_%i' % i):
                    num_test += 1

    X_train = np.empty(shape=(num_train, 16071 ,7))         #hardcoded
    y_train = np.empty(shape=(num_train, 1))                     #hardcoded
    X_test = np.empty(shape=(num_test, 16071, 7))           #hardcoded
    y_test = np.empty(shape=(num_test, 1))                       #hardcoded

    # Extract training and test sets from data dictionary
    idx = 0
    for subj in train:
        for i in xrange(1, 9):
            task_id = 'task_%i' % i
            if hasattr(data[subj], task_id):
                x = getattr(data[subj], task_id)
                x = x.reshape((1, x.shape[0], x.shape[1]))
                X_train[idx] = x
                y_train[idx] = int(data[subj].info['PD status'])
                idx += 1

    idx = 0
    for subj in test:
        for i in xrange(1, 9):
            task_id = 'task_%i' % i
            if hasattr(data[subj], task_id):
                x = getattr(data[subj], task_id)
                x = x.reshape((1, x.shape[0], x.shape[1]))
                X_test[idx] = x
                y_test[idx] = int(data[subj].info['PD status'])
                idx += 1
    
    return X_train, y_train, X_test, y_test

def normalize_data(X_train, X_test):
    """ Normalizes the following features in the training set to be in range [0, 1]. Then applies transformation 
        to test set. 
        - Azimuth
        - Altitude
        - Pressure

    Args:
        X_train: numpy array. 3D matrix containing our training sequences 
        X_test: numpy array. 3D array containing our test sequences
    """

def evaluate_model():
    """ Loads the data dictionary and then performs K-Fold cross validation to obtain the classification accuracy
        of our model.
    """
    data = process.load_dataset()
    feature_extraction(data)

    kf_splits = KFold(data)
    for i in xrange(len(kf_splits.train)):
        train = kf_splits.train[i]
        test = kf_splits.test[i]
        (X_train, y_train, X_test, y_test) = create_datasets(data, train, test)

        # Normalize data here

    return X_train, y_train, X_test, y_test

if __name__== '__main__':
    X_train, y_train, X_test, y_test = evaluate_model()