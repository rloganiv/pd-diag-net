"""Trains an LSTM model on the PaHaW dataset and evaluates using cross validation"""
import numpy as np
import process

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
                    x = np.vstack((x, np.zeros(shape=(max_len - x.shape[0], 8))))   # Adds extra time points
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
    pad_sequence(data, max_len=16071)   # maximum sequence length of data is 16071
    remove_time(data)

def normalize_data(data):
    pass

def evaluate_model():
    data = process.load_dataset()
    feature_extraction(data)

    # Cross validation here

if __name__== '__main__':
    data = evaluate_model()