from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

import pdb
import pickle
from process import Subject, PaHaWDataset, extract_datasets
import time

# Try to load pickled data, if it doesnt exist generate train and test dataset
# from scratch
try:
    with open('PaHaW/processed_data.pkl', 'rb') as pkl_file:
        data = pickle.load(pkl_file)
except:
    data = extract_datasets()

data.method = 'summary'

# Model definition
model = Sequential()
# model.add(LSTM(64, input_dim = 6, return_sequences = True))
#model.add(LSTM(100, input_dim = 7)) #, return_sequences = True))
#model.add(Dropout(0.20))
#model.add(LSTM(32))
#model.add(Dense(64, W_regularizer=l2(0.1), activation='relu'))
#model.add(Dropout(0.20))
model.add(Dense(1024, activation='relu', input_dim=9))
#model.add(Dropout(0.20))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile and fit model
# !!! WARNING: SPLIT THE DATA BEFORE TRAINING !!!
model.compile(loss='binary_crossentropy', optimizer='adagrad',
              metrics=['accuracy'])
model.fit(data.x, data.y, nb_epoch=100,
          batch_size=10)

# Save model
timestamp = time.strftime("%y-%m-%d_%H-%M")
model.save('model_%s.h5' % timestamp)
