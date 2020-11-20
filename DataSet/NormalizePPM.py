import sys, os
import numpy as np
import pickle
from pathlib import Path

'''
Normalize Structured Dataset

Args:
1 - Path to data to normalize 
2 - Path where store normalized data 
'''

def min_max_scaler(min_g, max_g, data):
	return [(matrix-min_g)/(max_g-min_g) for matrix in data]

ranges = pickle.load(open('../PreProcess/range', 'rb'))
min_ = ranges['min']
max_ = ranges['max']

path = sys.argv[1]
data = pickle.load(open(path, 'rb'))
scaled = min_max_scaler(min_, max_, data)

dest_path = sys.argv[2]
Path(dest_path).mkdir(parents=True, exist_ok=True)
pickle.dump(scaled, open(dest_path+'/scaled', 'wb'))
