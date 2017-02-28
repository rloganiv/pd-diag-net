"""Process the PaHaW dataset into NumPy arrays"""
import numpy as np
import pandas as pd
import pdb
import pickle
import scipy


# Constants
CORPUS_PATH = 'PaHaW/PaHaW_files/corpus_PaHaW.xlsx'
SVC_DIR = 'PaHaW/PaHaW_public'
IMG_DIM = (6000, 10000)
TINY_DIM = (300, 500)

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
    def __init__(self, subjects):
        self.subjects = subjects

        # Default values
        self._maxlen = None
        self._method = None
        self.subsample_rate = 100
        self.window_size = 100
        self.stride = 20

    def subsample(self, task):
        """Subsample rows from a task"""
        return task[::self.subsample_rate, :]

    def window(self, task):
        """Generate windows from a task"""
        n_windows = (task.shape[0] - self.window_size) // self.stride + 1
        return [
            task[(i*self.stride):(i*self.stride + self.window_size), :] for i in xrange(n_windows)
        ]

    def summarize(self, task):
        """Convert sequence of task values into summary statistics"""
        in_air = task[:, 3] == 0
        try:
            summary_vect = np.array([
                np.std(task[in_air, 13]), # Std. dev. of in air velocity
                np.min(task[in_air, 11]), # Min of vertical jerk
                np.std(task[in_air, 14]), # Std. dev. of in air accel.
                # Range of horizontal jerk
                np.max(task[in_air, 12]) - np.min(task[in_air, 12]),
                np.std(task[in_air, 15]), # Std. dev of in air jerk
                # Range of horizontal accel.
                np.max(task[in_air, 10]) - np.min(task[in_air, 10]),
                # Range of horizontal velocity.
                np.max(task[in_air, 8]) - np.min(task[in_air, 8]),
                # 75th percentile of on-surface horizontal velocity
                np.percentile(task[np.logical_not(in_air), 8], 0.75),
                np.min(task[in_air, 10]), # Min. in air vertical accel.
                # 99-1 percentile vertical velocity
                np.percentile(task[in_air, 8], 0.99) - np.percentile(task[in_air, 8], 0.01),
                # Mean velocity
                np.mean(task[in_air, 13]),
                # Mean altitutde velocity
                np.mean(task[in_air, 17]),
                # 99-1 percentile altitude velocity
                np.percentile(task[in_air, 17], 0.99) - np.percentile(task[in_air, 17], 0.01),
                # Std dev. altitude velocity
                np.std(task[in_air, 17])

            ])
            return summary_vect
        except:
            # If there are issues generating summary stats it is probably
            # because pen was never lifted from paper - e.g. in task 1. We will
            # just skip the task if this happens.
            pass

    def extract_imgs(self, task):
        # Initialize image arrays
        img_paper = np.zeros(IMG_DIM)
        img_air = np.zeros(IMG_DIM)
        # Extract coordinates
        x = task[:, 1]
        y = task[:, 0]
        idx = task[:, 3]
        # Upsample handwriting samples
        t = np.arange(task.shape[0])
        dt = np.linspace(0, task.shape[0], 100*task.shape[0])
        x = np.interp(dt, t, x).astype(np.int16)
        y = np.interp(dt, t, y).astype(np.int16)
        idx = np.interp(dt, t, idx).astype(np.int16)
        # Seperate on-paper and in-air samples
        on_paper = idx == 1
        in_air = idx == 0
        # Create thickened images
        n = 5
        for dx in xrange(-n, n+1):
            for dy in xrange(-n, n+1):
                img_paper[y[on_paper] + dy, x[on_paper] + dx] += 1
                img_air[y[in_air] + dy, x[in_air] + dx] += 1
        # Downsample images
        img_paper = img_paper[::20, ::20]
        img_air = img_air[::20, ::20]
        return img_paper, img_air

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
            if self.method is 'summary':
                tasks = [self.summarize(task) for task in
                         subject.task.itervalues()]
                tasks = [task for task in tasks if task is not None]
            if self.method is 'img':
                img_paper = np.zeros(TINY_DIM)
                img_air = np.zeros(TINY_DIM)
                for task in subject.task.itervalues():
                    dp, da = self.extract_imgs(task)
                    img_paper += dp
                    img_air += da
                tasks = [(img_paper, img_air)]
            x += tasks
            y += [subject.info['PD status']] * len(tasks)

        # Convert to arrays
        if self.method == 'summary':
            x = np.vstack(x)
        elif self.method == 'img':
            pdb.set_trace()
            on_paper = np.stack([task[0].reshape(1, TINY_DIM[0], TINY_DIM[1])  for task in x])
            in_air = np.stack([task[1].reshape(1, TINY_DIM[0], TINY_DIM[1]) for task in x])
            x = on_paper, in_air
        else:
            x = sequence.pad_sequences(x, maxlen=400, dtype='float32')

        y = np.array(y, dtype='float32')

        # Normalize x-values
        # x = x / np.max(x, axis=(0,1))

        # Assign attributes
        self.x = x
        self.y = y

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        if method in ['subsample', 'window', 'summary', 'img', None]:
            self._method = method
            self.update()
        else:
            raise ValueError('Invalid method')

    @property
    def maxlen(self):
        return self._maxlen

    @maxlen.setter
    def maxlen(self, maxlen):
        self._maxlen = maxlen
        self.update()

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


def parse_svc(path, i):
    """Processes the data in an .svc file to a numpy array

    For more information on the .svc format refer to the 'info.txt' file
    included with the dataset.

    Args:
        path: string. Path to the .svc file.

    Returns:
        np.array. Each row of the array is a sample. The columns represent:
            0: y-value,
            1: x-value,
            2: timestamp,
            3: button state,
            4: azimuth,
            5: altitude,
            6: pressure,
            7: y-velocity,
            8: x-velocity,
            9: y-acceleration,
            10: x-acceleration,
            11: y-jerk,
            12: x-jerk,
            13: velocity,
            14: acceleration,
            15: jerk,
            16: azimuth-velocity,
            17: altitude-velocity,
            18: pressure-velocity,
            19-26: One hot encoding of task

    """
    from sklearn.preprocessing import OneHotEncoder

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

    # remove time stamp
    # array = np.delete(array, 2, axis=1)
    n = array.shape[0]

    # position based velocity, acceleration, and jerk
    xy_vel = displace(array[:,0:2])
    xy_accel = displace(xy_vel)
    xy_jerk = displace(xy_accel)

    # magnitudes of previous measurements
    m_vel = np.linalg.norm(xy_vel, axis=1).reshape((n, 1))
    m_accel = np.linalg.norm(xy_accel, axis=1).reshape((n, 1))
    m_jerk = np.linalg.norm(xy_jerk, axis=1).reshape((n, 1))

    # rate of change of azimuth, altitude, and pressure
    aap_vel = displace(array[:,4:])

    # add task id as feature
    # print array.shape
    new_col = (i-1) * np.ones((n,1))
    enc = OneHotEncoder(n_values=8)
    # one_hot = np.asarray(enc.fit_transform(new_col).todense())

    out = np.concatenate((array, xy_vel, xy_accel, xy_jerk, m_vel, m_accel,
                          m_jerk, aap_vel), axis=1)
    return out


def displace(array):
    """Generates displacement vectors for columns in an array.

    Arg:
        array: np.array. Vectors to be displaced.
    """
    disp = np.zeros(shape=array.shape)
    disp[1:,:] = array[1:,:] - array[0:-1,:]
    return disp


def extract_datasets(test_fraction = 0.333):
    """Extracts the train and test datasets.

    Returns:
        train, test: PaHaWDataset() objects.
    """
    import random
    #train_subjects = []
    #test_subjects = []
    subjects = []

    # Extract data from corpus
    corpus = parse_corpus(CORPUS_PATH)

    # Build a Subject object for each row in the corpus and randomly assign to
    # train or test dataset
    for i, row in enumerate(corpus.iterrows()):
        subject = Subject()
        subject.info = row[1]
        subject.task = dict()
        # ID must be converted from int to fixed length string.
        subject_id = '%05d' % row[1].ID
        # Extract task data from SVC files
        for i in xrange(1, 9):
            try:
                svc_path = generate_svc_path(subject_id, i)
                task_data = parse_svc(svc_path, i)
                subject.task[i] = task_data
            except IOError:
                print 'Subject %s did not perform task %i' % (subject_id, i)
        # if random.random() < test_fraction:
        #     test_subjects.append(subject)
        # else:
        #     train_subjects.append(subject)
        subjects.append(subject)
    # Load subjects into datasets
    # train = PaHaWDataset(train_subjects)
    # test = PaHaWDataset(test_subjects)
    data = PaHaWDataset(subjects)

    # return train, test
    return data


if __name__ == '__main__':
    print "Extracting data"
    data = extract_datasets()
    print "Saving data"
    with open('PaHaW/processed_img_data.pkl', 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
    print "Done!"

