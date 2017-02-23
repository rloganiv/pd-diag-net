from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

import pickle
from process import Subject, PaHaWDataset, extract_datasets
import time

# Try to load pickled data, if it doesnt exist generate train and test dataset
# from scratch
try:
    with open('PaHaW/processed_data.pkl', 'rb') as pkl_file:
        train, test = pickle.load(pkl_file)
except:
    train, test = extract_datasets()

# Model definition
model = Sequential()
model.add(LSTM(100, input_dim = 6))
model.add(Dense(1, activation='sigmoid'))

# Compile and fit model
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(train.x, train.y, validation_data=(test.x, test.y), nb_epoch=3,
          batch_size=5)

# Save model
timestamp = time.strftime("%y-%m-%d_%H-%M")
model.save('model_%s.h5' % timestamp)
