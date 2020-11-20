import numpy as np
import pickle
from pathlib import Path
import matplotlib
import matplotlib.cm as cm

ranges = pickle.load(open('../../PreProcess/range', 'rb'))
min_g = ranges['min']
max_g = ranges['max']

''' Saving sensors data for Structured Dataset '''
data = pickle.load(open('sensors_data', 'rb'))
path = '../Structured/'
Path(path).mkdir(parents=True, exist_ok=True)
pickle.dump(data, open(path+'sensor', 'wb'))

''' Transform sensors data for RGB Dataset '''
sm = cm.ScalarMappable(cmap = 'viridis')
sm.set_clim(vmin = min_g, vmax = max_g)
rgbs = []
for elem in data:
    img = sm.to_rgba(elem, bytes = True)
    rgbs.append(img[:,:,:3])

path = '../RGB/'
Path(path).mkdir(parents=True, exist_ok=True)
pickle.dump(rgbs, open(path+'sensor', 'wb'))