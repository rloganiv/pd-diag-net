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
        task: dict. Processed .svc data for tasks, e.g. Subject.task[i]
            contains the numpy array of data corresponding to task i.
    """
    pass


class PaHaWDataset(object):
    """A dataset object"""
    def __init__(self, subjects, method=None, subsample_rate=100, window_size=100,
                 stride=25):
        self.subjects = subjects
        self.method = method
        self.subsample_rate = subsample_rate
        self.window_size = window_size
        self.stride = stride

    def subsample(self, task):
        """Subsample rows from a task"""
        return task[::self.subsample_rate, :]

    def window(self, task):
        """Generate windows from a task"""
        n_windows = (task.shape[0] - self.window_size) // self.stride + 1
        return [
            task[(i*self.stride):(i*self.stride + self.window_size), :] for i in xrange(n_windows)
        ]

    def update(self):
        from keras.preprocessing import sequence
        x = []
        y = []

        # Process the task data according to active method for each subject.
        # Append results to x and y
        for subject in self.subjects:
            if self.method == 'subsample':
                tasks = [self.subsample(task) for task in subject.task.itervalues()]
            if self.method == 'window':
                tasks = []
                for task in subject.task.itervalues():
                    tasks += self.window(task)
            if self.method is None:
                tasks = subject.task.values()
            x += tasks
            y += [subject.info['PD status']] * len(tasks)

        # Replace x and y attributes with updated numpy arrays
        self.x = sequence.pad_sequences(x)
        self.y = np.array(y)

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        if method in ['subsample', 'window', None]:
            self._method = method
            self.update()
        else:
            raise ValueError('Invalid method')

    @property
    def subsample_rate(self):
        return self._subsample_rate

    @subsample_rate.setter
    def subsample_rate(self, subsample_rate):
        if type(subsample_rate) is int:
            self._subsample_rate = subsample_rate
            self.update()
        else:
            raise ValueError('Subsample rate must be an int')

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, window_size):
        if type(window_size) is int:
            self._window_size = window_size
            self.update()
        else:
            raise ValueError('Window size must be an int')

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, stride):
        if type(stride) is int:
            self._stride = stride
            self.update()
        else:
            raise ValueError('Stride must be an int')


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
    df['PD status'] = df['PD status'].astype(int)
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
            0: y-displacement,
            1: x-displacement,
            2: button state,
            3: azimuth,
            4: altitude,
            5: pressure
    """
    # Open the file
    with open(path, 'r') as svc_file:
        samples = svc_file.readlines()

    # Extract the data
    data = []
    for sample in samples[1:]:
        values = [int(value) for value in sample.split()]
        data.append(values)
    array = np.array(data)

    # ---Process the data---

    # x,y coords to displacement
    array[1:, 0:2] -= array[0:-1, 0:2]
    array[0, 0:2] = 0

    # remove time stamp
    array = np.delete(array, 2, axis=1)

    return array


def extract_datasets(test_fraction = 0.333):
    """Extracts the train and test datasets.

    Returns:
        train, test: PaHaWDataset() objects.
    """
    import random
    train_subjects = []
    test_subjects = []

    # Extract data from corpus
    corpus = parse_corpus(CORPUS_PATH)

    # Build a Subject object for each row in the corpus and randomly assign to
    # train or test dataset
    for row in corpus.iterrows():
        subject = Subject()
        subject.info = row[1]
        subject.task = dict()
        # ID must be converted from int to fixed length string.
        subject_id = '%05d' % row[1].ID
        # Extract task data from SVC files
        for i in xrange(1, 9):
            try:
                svc_path = generate_svc_path(subject_id, i)
                task_data = parse_svc(svc_path)
                subject.task[i] = task_data
            except IOError:
                print 'Subject %s did not perform task %i' % (subject_id, i)
        if random.random() < test_fraction:
            test_subjects.append(subject)
        else:
            train_subjects.append(subject)
    # Load subjects into datasets
    train = PaHaWDataset(train_subjects)
    test = PaHaWDataset(test_subjects)

    return train, test


if __name__ == '__main__':
    train, test  = extract_datasets()
    with open('PaHaW/processed_data.pkl', 'wb') as pkl_file:
        pickle.dump((train, test), pkl_file)

