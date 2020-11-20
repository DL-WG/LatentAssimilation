import sys, os
import numpy as np
import pickle
from pathlib import Path

'''
Normalize RGB Dataset

Args:
1 - Path to data to normalize (es: '../DataSet/RGB/data')
2 - Path where store normalized data (es: 'RGB/')
'''

path = sys.argv[1]
rgb = pickle.load(open(path, 'rb'))
scaled = [x/255 for x in rgb]

dest_path = sys.argv[2]
Path(dest_path).mkdir(parents=True, exist_ok=True)
pickle.dump(scaled, open(dest_path+'/scaled', 'wb'))