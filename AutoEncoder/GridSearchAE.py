import numpy as np
import sys, os
import datetime, time
import pickle
import importlib.util
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from pathlib import Path

'''
Grid Search for Autoencoder
Args:
1 - Path to module that contains the structure of the autoencoder (es: Structured/LS1/GridSearch.py)
2 - Path to train dataset (es: ../DataSet/Structured)
3 - Latent Space (es: 1)
4 - Path where results will be stored (es: Structured/LS1/)
'''

''' Import the Config module and the module that contains the structure of the autoencoder'''
modules = ['../Config.py', sys.argv[1]]
for path_module in modules:
	spec = importlib.util.spec_from_file_location("models", path_module)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

strategy = tf.distribute.MirroredStrategy()

path = sys.argv[2]
train = pickle.load(open(path+'/train', 'rb'))

rows = train[0].shape[0]
columns  = train[0].shape[1]
channels = 1 if (len(train[0].shape) < 3) else train[0].shape[2]

''' Reshape train for autoencoder '''
train = (np.array(train)).reshape(len(train), rows, columns, channels)

latent_space = int(sys.argv[3])

dest_path = sys.argv[4] + '/' + module.PATH
Path(dest_path).mkdir(parents=True, exist_ok=True)

act = ['relu', 'elu']
n_fil = [16, 32, 64]
batch = [16, 32, 64]
epochs = [250, 300, 400]

input_shape = (rows, columns, channels)
X_input = Input(input_shape)

models = os.listdir(dest_path)

KF = KFold(n_splits = 5, shuffle = True, random_state = 1)
for a in act:
	for f in n_fil:
		for b in batch:
			for e in epochs:
				name= str(f) + '-' + a + '-' + str(b) + '-' + str(e)
				if 'result-' + name not in models:
					try:
						CVResults = {}
						
						i=1
						print(name, ' CV: ', str(i) )

						for train_index, test_index in KF.split(train):
							x_train = train[train_index]
							x_test = train[test_index]
							
							start = time.time()

							with strategy.scope():
								autoencoder = module.model(X_input, f, a, latent_space)
								autoencoder.compile(optimizer='adam', loss='mse', metrics = ['mse','mae'])

							history = autoencoder.fit(x_train, x_train, epochs = e, batch_size = b, validation_data = (x_test, x_test), verbose = 1)
							loss = autoencoder.evaluate(x_test, x_test)

							end = time.time()

							result = {
								'history' : history.history,
								'loss' : loss[0],
								'mse' : loss[1],
								'mae' : loss[2],
								'time' : end-start
							}

							CVResults[i] = result
							i+=1

						pickle.dump(CVResults, open(dest_path + 'result-'+str(name), 'wb'))
					except ValueError:
						pass
				else:
					print(name)

