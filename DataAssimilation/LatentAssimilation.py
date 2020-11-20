import numpy as np
import sys
import pickle
import KalmanFilter as KF
import datetime, time
import importlib.util
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from texttable import Texttable

'''
Args:
1 - Path to autoencoder model 
2 - Path to LSTM model 
3 - Path to test data 
4 - Path to sensor data 
'''

spec = importlib.util.spec_from_file_location("models", '../AutoEncoder/AutoEncoder.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

spec = importlib.util.spec_from_file_location("Utils", '../LSTM/Utils.py')
Utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Utils)

latent_space = int(sys.argv[1])

''' Reading autoencoder model '''
print('Loading autoencoder')
autoencoder = module.Autoencoder(latent_space)
autoencoder.load_model(sys.argv[2])
encoder, _ = autoencoder.encoder()
decoder = autoencoder.decoder()

''' Reading lstm model '''
print('Loading LSTM')
path_lstm = sys.argv[3]
lstm = tf.keras.models.load_model(path_lstm)

''' Reading test data'''
print('Loading test data')
data = pickle.load(open(sys.argv[4], 'rb'))

rows = data[0].shape[0]
columns = data[0].shape[1]
channels = 1 if (len(data[0].shape) < 3) else data[0].shape[2]
data = np.array(data).reshape(len(data), rows, columns, channels)

''' Reading sensors data'''
print('Loading sensors data')
sensors_data = pickle.load(open(sys.argv[5], 'rb'))

''' Reading index position of sensors data in test '''
pos = pickle.load(open('SensorsData/pos_sensors_test', 'rb'))
n_steps = lstm.layers[0].input_shape[1]

''' Encoding Data '''
data = encoder.predict(data)

''' Reshaping '''
data = np.array(data).reshape(len(data), data[0].shape[0])

''' Preparing data for LSTM '''
test_x, test_y = Utils.split_sequences(data, n_steps)

''' Sequence we are interested in '''
positions = [x-n_steps for x in pos if (x-n_steps)>=0]
to_predict = [test_x[i] for i in positions]

sensors_data = sensors_data[-len(positions):]
sensors_data = np.array(sensors_data).reshape(len(sensors_data), rows, columns, channels)
sensors_y = encoder.predict(sensors_data)

print('Start Assimilation')

table = Texttable()
table.set_cols_dtype(['t', 'e', 'e', 'e', 'e'])
table.add_row(['R', 'Conv', '0.01', '0.001', '0.0001'])

''' Predicting '''
predicted = lstm.predict(np.array(to_predict).reshape(len(to_predict), to_predict[0].shape[0], to_predict[0].shape[1])) #ndarray (7,1)

I = np.identity(latent_space)
R = [KF.covariance_matrix(np.array(sensors_y).T), I * 0.01, I * 0.001, I * 0.0001]
C = I
P = KF.covariance_matrix(np.array(predicted).T)

results = ['MSE']
times = ['Time']

mse = tf.keras.losses.MeanSquaredError()

for r in R:

    start = time.time()
    K = KF.KalmanGain(P, C, r)
    updated = [KF.update_prediction(predicted[i].reshape(latent_space,), K, C, sensors_y[i]) for i in range(len(predicted))] #compute only the updated state
    end = time.time()   

    results.append((mse(sensors_y,updated)).numpy())
    times.append(end-start)

table.add_row(results)
table.add_row(times)

print('MSE without Assimilation in the Latent Space: ', (mse(sensors_y, predicted)).numpy())

print(table.draw())