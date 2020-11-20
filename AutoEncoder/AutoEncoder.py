import numpy as np
import sys, os
import datetime, time
import pickle
import inspect
import tensorflow as tf
import importlib.util
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.utils import shuffle
from pathlib import Path

class Autoencoder:

    def __init__(self, latentspace):
        print(latentspace)
        self.ls = latentspace

    ''' Load the model '''
    def load_model(self, filename):
        model = tf.keras.models.load_model(filename)
        self.model = model

    def get_model(self):
        return self.model

    ''' Fit and return the model '''
    def fit_model(self, filename, path_data, path_dest, n_fil, act, batch, epc):

        modules = ['../Config.py', filename]
        for path_module in modules:
            spec = importlib.util.spec_from_file_location("models", path_module)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        strategy = tf.distribute.MirroredStrategy()

        data = pickle.load(open(path_data, 'rb'))
        data = shuffle(data, random_state = 1)
        
        rows = data[0].shape[0]
        columns  = data[0].shape[1]
        channels = 1 if (len(data[0].shape) < 3) else data[0].shape[2]
        data = np.array(data).reshape(len(data), rows, columns, channels)

        input_shape = (rows, columns, channels)
        X_input = Input(input_shape)
        with strategy.scope():
            autoencoder = module.model(X_input, n_fil, act, self.ls)
            autoencoder.compile(optimizer='adam', loss='mse', metrics = ['mse','mae'])

        history = autoencoder.fit(data, data, epochs = epc, batch_size = batch, shuffle = True, validation_split = 0.2, verbose = 1)
        
        if path_dest != False:
            Path(path_dest).mkdir(parents=True, exist_ok=True)
            name = str(n_fil) + '-' + act + '-' + str(batch) + '-' + str(epc) + '-' + str(self.ls)
            autoencoder.save(path_dest + 'model-' + name + '.h5')            

        self.model = autoencoder
        return autoencoder
        

    ''' Build and return the encoder '''
    def encoder(self):
        X_input = Input((self.model.layers[0].input.shape)[-3:])
        encoder = X_input
        for i in range(1, len(self.model.layers)):
            encoder = self.model.layers[i](encoder)
            if(self.model.layers[i].output.shape[1] == int(self.ls)):
                print('IN')
                break;
        encoder = Model(inputs = X_input, outputs = encoder)
        return encoder, i

    ''' Build and return the decoder '''
    def decoder(self):
        e, layer = self.encoder()
        X_input = Input((self.ls))
        decoder = X_input
        for i in range(layer+1, len(self.model.layers)):
            decoder = self.model.layers[i](decoder)
        decoder = Model(inputs = X_input, outputs = decoder)
        return decoder

    def get_ls(self):
        return self.ls

'''
Fit the model given parameters
Args:
1 - Latent space
2 - Path to module that contains the structure of the model
3 - Path to data
4 - Path where the model will be stored
5 - Number of filters
6 - Activation funztion
7 - Bath size
8 - Number of epochs
'''
if __name__ == '__main__':
    if(len(sys.argv)>2):

        ae = Autoencoder(int(sys.argv[1]))        
        ae.fit_model(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], int(sys.argv[7]), int(sys.argv[8]))
