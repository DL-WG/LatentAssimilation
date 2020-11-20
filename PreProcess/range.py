import numpy as np
import sys, os
import vtktools
import pickle
import pandas as pd

'''
Compute the global minimum and global maximum of tracer values between both datasets and sensors data.
The values are in PPM scale.
'''

path = './Room/'

min_ = 1
max_ = 0

ranges= {}

for i in range(2501):
    filename = 'LSBUIndoor_'+str(i)+'_ext.vtu'
    try:
        ob = vtktools.vtu(path+filename)
        values = ob.GetScalarRange('Tracer')
        min_ = np.min([min_, values[0]])
        max_ = np.max([max_, values[1]])
    except:
        print('Error: ', filename)
        pass

    if(i%100 == 0):
        print(i)

ranges['room'] = { 
        'min' : min_,
        'max' : max_
    }

structured = []
for i in range(2501):
    try:
        structured.append(pickle.load(open('Structured/'+str(i), 'rb')))
    except:
        print('Error: ', i)
        pass

min_str = np.min([np.min(x) for x in structured])
max_str = np.max([np.max(x) for x in structured])
ranges['structured'] = { 
        'min' : min_str,
        'max' : max_str
    }

sensors = pd.read_excel('../DataAssimilation/SensorsData/CO2_Maddalena.xlsx', index_col = 'Time')
sensors.drop(["Fluidity Time (sec)", 'Outdoor'], axis = 1, inplace = True)
min_sensors = sensors.min().min()
max_sensors = sensors.max().max()
ranges['sensors'] = { 
        'min' : min_sensors,
        'max' : max_sensors
    }

coeff = 5.5e5
min_g = np.min([min_*coeff, min_str, min_sensors])
max_g = np.max([max_*coeff, max_str, max_sensors])

ranges['min'] = min_g
ranges['max'] = max_g

print('Global min: ', min_g)
print('Global max: ', max_g)

pickle.dump(ranges, open('range','wb'))