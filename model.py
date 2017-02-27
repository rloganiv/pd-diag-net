from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import numpy as np
import pdb
import pickle
from process import Subject, PaHaWDataset, extract_datasets
import time

K.set_image_dim_ordering('th')

batch_size = 5
nb_folds = 10
nb_epoch = 30

img_rows = 300
img_cols = 500

mdl_type = 'air'

# Try to load pickled data, if it doesnt exist generate train and test dataset
# from scratch
print "Loading data"
try:
    with open('PaHaW/processed_img_data.pkl', 'rb') as pkl_file:
        data = pickle.load(pkl_file)
except:
    data = extract_datasets()

img_paper, img_air = data.x
print img_paper.shape

if mdl_type == 'air':
    x = img_air
elif mdl_type == 'paper':
    x = img_paper

# Model definition
def make_model():
    model = Sequential()
    model.add(Convolution2D(8, 4, 4, input_shape = (1, img_rows, img_cols),
                            W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 4, 4, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.50))

    model.add(Flatten(input_shape = (1, img_rows, img_cols)))
    model.add(Dense(8, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

# Image generator
datagen = ImageDataGenerator(
    rotation_range=2.5,
    width_shift_range=0.025,
    height_shift_range=0.025,
    zoom_range=0.025
)

if __name__ == '__main__':
    print 'Starting main loop'
    # 10 fold cross validation
    n_samples = data.y.shape[0]
    fold_size = n_samples // nb_folds
    idx = np.arange(n_samples)
    np.random.shuffle(idx)

    hists = {}

    timestamp = time.strftime("%y-%m-%d_%H-%M")

    for k in xrange(nb_folds):
        print "On fold %i" % k
        valid_idx = idx[k*fold_size:(k+1)*fold_size]
        train_idx = np.setdiff1d(idx, valid_idx)

        x_train = x[train_idx]
        x_valid = x[valid_idx]
        y_train = data.y[train_idx]
        y_valid = data.y[valid_idx]

        #fit feature generator
        datagen.fit(x_train)

        print 'Training model'
        model = make_model()
        hist = model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            samples_per_epoch = n_samples,
            validation_data = (x_valid, y_valid),
            nb_epoch=nb_epoch)
        model.save('saved_models/model_%s_fold_%i_%s.h5' % (mdl_type, k, timestamp))
        hists[k] = hist.history

    with open('saved_models/model_%s_hists_%s.pkl' % (mdl_type, timestamp), 'wb') as pkl_file:
        pickle.dump(hists, pkl_file)

