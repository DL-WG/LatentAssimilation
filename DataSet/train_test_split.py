import numpy as np
import pandas as pd
import sys, os
import pickle

'''
Train and Test preparation.

Args:
1 - Path to scaled data 
'''

timestep = pickle.load(open('../PreProcess/time_step', 'rb'))
sensors_data = (pd.read_excel('../DataAssimilation/SensorsData/Observations.xlsx', index_col = 'Time'))['Fluidity Time (sec)']

#compute the timesteps of the sensors data
sensors_ts =sorted(list(set([timestep.index(min(timestep, key=lambda time:abs(time-x))) for x in sensors_data])))
print('Sensors data: ',len(sensors_ts))

#remove the sensors time step from original data
index = [item for item in np.arange(0,2500,1) if item not in sensors_ts]

jump = 2

x_test = []
for i in range((jump*2)+1, len(index), jump*3):
    x_test.append(index[i])
x_test = sorted(list(set(x_test) | set(sensors_ts))) #union of test set and sensors time step and sorting

x_train = [i for i in index if i not in x_test]

print('Train: ', len(x_train))
print('Test: ', len(x_test))

path = sys.argv[1]
data = pickle.load(open(path+'/scaled','rb'))
pickle.dump([data[x] for x in x_train], open(path + '/train','wb'))
pickle.dump([data[x] for x in x_test], open(path + '/test','wb'))

''' Store the position of sensor data in the test set'''
pos = [x_test.index(x) for x in sensors_ts]
pickle.dump(pos, open('../DataAssimilation/SensorsData/pos_sensors_test','wb'))