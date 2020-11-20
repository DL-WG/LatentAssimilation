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
Grid Search for LSTM
Args:
1 - Path to train and validation sets 
2 - Latente Space 
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

dest_path = sys.argv[3] + '/ResultGS/'
Path(dest_path).mkdir(parents=True, exist_ok=True)

neurons = [30, 50, 70]
act = ['relu', 'elu']
n_step = [3,5,7]
batch = [16, 32, 64]
epochs = [200, 300, 400]

n_reps = 5

for n in neurons:
    for a in act:
        for s in n_step:
            lstm_train, train_y = Utils.split_sequences(train, s)
            lstm_val, val_y = Utils.split_sequences(val, s)
            for e in epochs:
                for b in batch:

                    name = str(s)+'-'+a+'-'+str(n)+'-'+str(e)+'-'+str(b)
                    print(name)

                    model_res = {}
                    
                    try:
                        for i in range(1,n_reps+1):
                            result = {}  
                            start = time.time()

                            with strategy.scope():               
                                model = Sequential()
                                model.add(LSTM(n, activation = a, input_shape = (s, latent_space)))
                                model.add(Dense(latent_space))
                                model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

                            history = model.fit(lstm_train, train_y, epochs=e, batch_size=b, validation_data=(lstm_val, val_y), verbose=1, shuffle=False)
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
                            
                            model_res[i] = result

                        pickle.dump(model_res, open(dest_path + 'result-'+name, 'wb'))
                    except:
                        pass