from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, ZeroPadding2D

def model(X_input, n_fil, act, LS):
	encoder = Conv2D(filters=n_fil, kernel_size=(3,3), padding = 'same', activation = act)(X_input)
	encoder = MaxPooling2D(2,2)(encoder)
	encoder = Conv2D(filters=n_fil, kernel_size=(3,3), padding = 'same', activation = act)(encoder)
	encoder = MaxPooling2D(2,2)(encoder)
	encoder = Conv2D(filters=n_fil, kernel_size=(3,3), padding = 'same', activation = act)(encoder)
	encoder = MaxPooling2D(2,2)(encoder)
	encoder = Conv2D(filters=1, kernel_size=(3,3), padding = 'same', activation = act)(encoder)
	encoder = MaxPooling2D(2,2)(encoder)
	encoder = Flatten()(encoder)
	encoder = Dense(LS)(encoder)

	decoder = Dense(165)(encoder)
	decoder = Reshape((11,15, 1))(decoder)
	decoder = Conv2D(filters=n_fil, kernel_size=(3,3), padding = 'same', activation = act)(decoder)
	decoder = UpSampling2D(size = (2,2))(decoder)
	decoder = Conv2D(filters=n_fil, kernel_size=(3,3), padding = 'same', activation = act)(decoder)
	decoder = UpSampling2D(size = (2,2))(decoder)
	decoder = ZeroPadding2D(((0,1),(1,1)))(decoder)
	decoder = Conv2D(filters=n_fil, kernel_size=(3,3), padding = 'same', activation = act)(decoder)
	decoder = UpSampling2D(size = (2,2))(decoder)
	decoder = ZeroPadding2D(((0,0),(0,1)))(decoder)
	decoder = Conv2D(filters=n_fil, kernel_size=(3,3), padding = 'same', activation = act)(decoder)
	decoder = UpSampling2D(size = (2,2))(decoder)
	decoder = Conv2D(filters=X_input.shape[3], kernel_size=(3,3), padding = 'same', activation = act)(decoder)
	
	autoencoder = Model(inputs = X_input, outputs = decoder)
	return autoencoder

PATH = 'ResultGS/'