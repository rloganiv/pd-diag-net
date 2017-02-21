"""Process the PaHaW dataset into NumPy arrays"""

import numpy as np
import pandas as pd
import pickle

# Constants
CORPUS_PATH = 'PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
SVC_DIR = 'PaHaW/PaHaW_public'


class Subject(object):
    """Stores the subject level data

    Attributes:
        info: Contains the subject charactersitics from the 'corpus_PaHaW.xlsx'
            spreadsheet.
        task_1: Processed .svc data for task 1.
        ...
        task_8: Processed .scv data for task 8.
    """
    pass


def parse_corpus(path):
    """Processes the data in the 'corpus_PaHaW.xlsx' spreadsheet.

    Args:
        path: string. Path to the spreadsheet.

    Returns:
        pandas.dataframe. Contains the processed spreadsheet data.
    """
    df = pd.read_excel(path)

    # Replace PD status with a binary variable.
    df['PD status'] = df['PD status'] == 'ON'
    # --- ADD OTHER PROCESSING STEPS HERE ---

    return df


def generate_svc_path(subject_id, task_index):
    """Generates the path to the .svc file for a given subject and task.

    Args:
        subject_id: int. ID of the subject.
        task_index: int. Index of the task.

    Returns:
        string. Path to the .svc file.
    """
    return "%s/%s/%s__%i_1.svc" % (SVC_DIR, subject_id, subject_id, task_index)


def parse_svc(path):
    """Processes the data in an .svc file to a numpy array

    For more information on the .svc format refer to the 'info.txt' file
    included with the dataset.

    Args:
        path: string. Path to the .svc file.

    Returns:
        np.array. Each row of the array is a sample. The columns represent:
            0: y-coordinate,
            1: x-coordinate,
            2: time stamp,
            3: button state,
            4: azimuth,
            5: altitude,
            6: pressure
    """
    # Open the file
    with open(path, 'r') as svc_file:
        samples = svc_file.readlines()

    # Process the data
    data = []
    for sample in samples[1:]:
        values = [int(value) for value in sample.split()]
        data.append(values)
    return np.array(data)


def load_dataset():
    """Builds the PaHaW dataset.

    Returns:
        dict. A dictionary whose keys are subject id's and whose values are the
            corresponding Subject objects. See the process.Subject() class
            docstring for more info.
    """
    # Store all subjects in a dictionary
    dataset = {}

    # Create a subject for each row in the corpus
    corpus = parse_corpus(CORPUS_PATH)
    for row in corpus.iterrows():
        subject = Subject()
        subject.info = row[1]
        # ID must be converted from int to fixed length string.
        subject_id = '%05d' % row[1].ID
        for i in xrange(1, 9):
            try:
                svc_path = generate_svc_path(subject_id, i)
                svc_data = parse_svc(svc_path)
                attr = 'task_%i' % i
                setattr(subject, attr, svc_data)
            except IOError:
                print 'Subject %s did not perform task %i' % (subject_id, i)
        dataset[subject_id] = subject

    return dataset

