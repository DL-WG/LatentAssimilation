import matplotlib
import numpy as np
import math
import sys, os
import pickle
import matplotlib.cm as cm
from pathlib import Path

'''
Cut the RGB elements in such a way they have shape (180,250,4).
The new RGB elements will be stored in RGB_cut folder.
'''

''' Fill the background pixels with the values of the pixels in the previous rows '''
def fill_cells(matrix, start, end, increment):
    for i in range(start, end, increment):
        matrix[:,i+increment] = [matrix[j, i] if (matrix[j,i+increment]==background).all() else matrix[j,i+increment] for j in range(len(matrix[:,i+increment]))]

input_dim = (215, 260) #RGB dimension
output_dim = (180, 250)
rows = int((input_dim[0] - output_dim[0])/2) #number of rows to cut
cols = int((input_dim[1] - output_dim[1])/2) #number of columns to cut

print('Rows: ', rows)
print('Columns: ', cols)

path = 'RGB/'
dest = 'RGB_cut/'
Path(dest).mkdir(parents=True, exist_ok=True)

background = np.array([76, 76, 76,  0])

print('Start')
for filename in range(2501):
    try:
        rgb = pickle.load(open(path+str(filename),'rb'))
        matrix = rgb[rows:rows+output_dim[0], cols:cols+output_dim[1]]

        c = 5
        fill_cells(matrix,output_dim[1]-1, 0, -1)
        fill_cells(matrix,output_dim[1] - c, output_dim[1]-1 ,1)

        pickle.dump(matrix, open(dest+str(filename), 'wb'))
    except:
        print('Error: ',filename)
        pass

    if filename%100==0:
    	print(filename)