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
from pathlib import Path

'''
Args:
1 - Path to train and validation datasets 
2 - Latent Space 
3 - Path where results will be stored 
'''

spec = importlib.util.spec_from_file_location("models", '../Config.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
strategy = tf.distribute.MirroredStrategy()

''' Read data '''
path = sys.argv[1]
train = pickle.load(open(path+'/train', 'rb'))
val = pickle.load(open(path+'/validation', 'rb'))

train = np.array(train).reshape(len(train), train[0].shape[0])
val = np.array(val).reshape(len(val), val[0].shape[0])

latent_space = int(sys.argv[2])

dest_path = sys.argv[3] + '/ResultLayers/'
Path(dest_path).mkdir(parents=True, exist_ok=True)

neurons = 30
act = 'relu'
n_step = 3
opt = 'adam'
batch = 32

lstm_train, train_y = Utils.split_sequences(train, n_step)
lstm_val, val_y = Utils.split_sequences(val, n_step)

layers=np.arange(1,6,1)
n_reps = 5

for l in layers:

    try:             
        model_res = {}

        for rep in range(1,n_reps+1):
            result = {}    

            with strategy.scope():           
                model = Sequential()
                if l == 1:
                    model.add(LSTM(neurons, activation = act, input_shape = (n_step,latent_space)))
                else:
                    model.add(LSTM(neurons, activation = act, return_sequences = True, input_shape = (n_step,latent_space)))                            
                    for i in range(2, l):
                        model.add(LSTM(neurons, activation = act, return_sequences = True))                            
                    model.add(LSTM(neurons, activation = act))
                model.add(Dense(latent_space))
                model.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae'])
            
            start = time.time()

            history = model.fit(lstm_train, train_y, epochs=300, batch_size=batch, validation_data=(lstm_val, val_y), verbose=1, shuffle=False)
            metrics = model.evaluate(lstm_val, val_y)

            end = time.time()

            print('Metrics: ',metrics)
            print('Time: ',end-start)

            result = {
                'history' : history.history,
                'mse' : metrics[1],
                'mae' : metrics[2],
                'time' : end-start
            }
            
            model_res[rep] = result
            
        pickle.dump(model_res, open(dest_path + 'result-'+str(l), 'wb'))
    except:
        pass