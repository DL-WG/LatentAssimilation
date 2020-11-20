import numpy as np
import pickle
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt

def interp(array, method_):
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    #mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method=method_)
    return GD1

room_shape = (530, 711) #cm

sensor_1 = (room_shape[0]-100, 10) 
sensor_2 = (room_shape[0]-430, 10) 
sensor_3 = (room_shape[0]-170, 700) 
sensor_4 = (room_shape[0]-10, 414) 
sensor_5 = (room_shape[0]-520, 314) 
sensor_6 = (room_shape[0]-453, 617) 
sensor_7 = (room_shape[0]-265, 414) 
coordinates = [sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7]

matrix_shape = (180, 250)
pixels = (matrix_shape[0]/room_shape[0], matrix_shape[1]/room_shape[1]) #number of pixel for 1 cm
print('Pixel per cm: ',pixels)

radius = 30 #cm
radius_pixels = (int(pixels[0] * radius), int(pixels[1] * radius)) #number of pixel to fill around the sensor position
print('Radius Pixel: ',radius_pixels)

coo_pixels = []
for c in coordinates:
    coo_pixels.append((int(pixels[0]*c[0]),int(pixels[1]*c[1])))

sensors_ts = len(pickle.load(open('pos_sensors_test', 'rb')))
df = pd.read_excel('Observations.xlsx', index_col = 'Time')
df = df[:sensors_ts]
df.drop(['Fluidity Time (sec)', 'Outdoor'], axis = 1, inplace=True)

sensor_data = [np.full(matrix_shape, np.nan) for i in range(len(df))]

print('Time steps: ',len(sensor_data))

print('Expanding sensors values')
for i in range(len(coo_pixels)):
    x = np.max([0,coo_pixels[i][0] - radius_pixels[0]])
    y = np.max([0, coo_pixels[i][1] - radius_pixels[1]])
    for m in range(len(sensor_data)):
        sensor_data[m][x:x+(2*radius_pixels[0]), y:y+(2*radius_pixels[1])] = df.iloc[m,i]

print('Linear Interpolation..')
data = []
for elem in sensor_data:
    res = interp(elem, 'linear')
    res = interp(res, 'nearest')
    data.append(res)

pickle.dump(data, open('sensors_data', 'wb'))