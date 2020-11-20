import numpy as np
import pandas as pd
import sys, os
import pickle
from pathlib import Path
import importlib.util

'''
Split the data into train and validation sets and encode the data.
We select for train set two consecutive time step and one for validation.

Args:
1 - Path to data
2 - Path to autoencoder model
3 - Latent space
4 - Path where data will be stored
'''
modules = ['../Config.py', '../AutoEncoder/AutoEncoder.py']
for path_module in modules:
	spec = importlib.util.spec_from_file_location("models", path_module)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

data = pickle.load(open(sys.argv[1], 'rb'))

jump = 2

train_lstm = []
for i in range(0,len(data),jump+1):
    train_lstm.extend([data[i], data[i+1]])

val_lstm = []
for i in range(jump, len(data), jump*3):
    val_lstm.append(data[i])

print('Train: ', len(train_lstm))
print('Validation: ', len(val_lstm))

rows = data[0].shape[0]
columns = data[0].shape[1]
channels = 1 if (len(data[0].shape) < 3) else data[0].shape[2]

ae = module.Autoencoder(sys.argv[3])
ae.load_model(sys.argv[2])
encoder, _ = ae.encoder()

train_lstm = np.array(train_lstm).reshape(len(train_lstm), rows, columns, channels)
val_lstm = np.array(val_lstm).reshape(len(val_lstm), rows, columns, channels)

encoded_train = encoder.predict(train_lstm)
encoded_val = encoder.predict(val_lstm)

print('Encoded Train: ', len(encoded_train), ' Shape: ', encoded_train[0].shape)
print('Encoded Validation: ', len(encoded_val), ' Shape: ', encoded_val[0].shape)

print('Saving')
path_dest = sys.argv[4]
Path(path_dest).mkdir(parents=True, exist_ok=True)
pickle.dump(encoded_train, open(path_dest + '/train','wb'))
pickle.dump(encoded_val, open(path_dest + '/validation','wb'))