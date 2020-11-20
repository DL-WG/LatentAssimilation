import numpy as np
import sys, os
import datetime, time
import pickle
import inspect
import importlib.util
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from sklearn.model_selection import KFold
from pathlib import Path

'''
Args:
1 - Path to module that contains models
2 - Path to train dataset
3 - Latent Space 
4 - Path where results will be stored 
'''

''' Import modules '''
modules = ['../Config.py', sys.argv[1]]
for path_module in modules:
	spec = importlib.util.spec_from_file_location("models", path_module)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

strategy = tf.distribute.MirroredStrategy()

models = [obj for name,obj in inspect.getmembers(module) 
                     if (inspect.isfunction(obj) and 
                         name.startswith('model') )]

path = sys.argv[2]
train = pickle.load(open(path+'/train', 'rb'))

rows = train[0].shape[0]
columns  = train[0].shape[1]
channels = 1 if (len(train[0].shape) < 3) else train[0].shape[2]

train = (np.array(train)).reshape(len(train), rows, columns, channels)

input_shape = (rows, columns, channels)
X_input = Input(input_shape)

latent_space = int(sys.argv[3])
dest_path = sys.argv[4] + '/' + module.PATH
Path(dest_path).mkdir(parents=True, exist_ok=True)

act = 'relu'
n_fil = 32
o = 'adam'
e = 300
b = 32

done = os.listdir(dest_path)

KF = KFold(n_splits = 5, shuffle = True, random_state = 1)
for model in models:
	model_n = model.__name__.split('model')[1]
	if 'result-'+model_n not in done:
		try:

			print('START: ', model_n)
			CVResults = {}

			i=1
			for train_index, test_index in KF.split(train):
				x_train = train[train_index]
				x_test = train[test_index]
				
				start = time.time()

				with strategy.scope():
					autoencoder = model(X_input, n_fil, act, latent_space)
					autoencoder.compile(optimizer=o, loss='mse', metrics = ['mse','mae'])

				history = autoencoder.fit(x_train, x_train, epochs = e, batch_size = b, validation_data = (x_test, x_test), verbose = 1)
				metrics = autoencoder.evaluate(x_test, x_test)

				end = time.time()
				
				result = {
					'history' : history.history,
					'loss' : metrics[0],
					'mse' : metrics[1],
					'mae' : metrics[2],
					'time' : end-start
				}

				CVResults[i] = result
				i+=1
		
		except:
				pass

		pickle.dump(CVResults, open(dest_path + 'result-'+str(model_n), 'wb'))
