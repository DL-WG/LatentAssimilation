import numpy as np
import sys, os
import datetime, time
import pickle
import importlib.util
import tensorflow as tf
import Utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

train = pickle.load(open(sys.argv[1] + './train', 'rb'))
train = np.array(train).reshape(len(train), train[0].shape[0])

latent_space = int(sys.argv[2])

n_step = int(sys.argv[3])
act = sys.argv[4]
neurons = int(sys.argv[5])
epc = int(sys.argv[6])
batch = int(sys.argv[7])

lstm_train, train_y = Utils.split_sequences(train, n_step)

model = Sequential()
model.add(LSTM(neurons, activation = act, input_shape = (n_step, latent_space)))
model.add(Dense(latent_space))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

history = model.fit(lstm_train, train_y, epochs=epc, batch_size=batch, verbose=1, shuffle=False)

name = str(n_step)+'-'+act+'-'+str(neurons)+'-'+str(epc)+'-'+str(batch)
model.save(sys.argv[8] + './model-' + name + '.h5')