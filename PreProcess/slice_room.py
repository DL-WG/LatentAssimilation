import sys, os
import pyvista as pv
import numpy as np
import pickle
from pathlib import Path

'''
Given the mesh of the room for the first 2500 timestep, compute the slice at half of the height of the room
'''

path = './Mesh/'
dest = './Slice/'
Path(dest).mkdir(parents=True, exist_ok=True)

for i in range(2501):
	try:
		filename = 'mLSBUIndoor_'+str(i)+'_ext.vtu'
		mesh = pv.read(path+filename)
		single_slice = mesh.slice(normal=[0, 0, 1], origin=[225.31896209716797, 66.47050094604492, 7.289999961853027])
		name_slice = filename.split('m')[1].split('_ext')[0]
		single_slice.save(dest + 's' + name_slice + '.vtk')

		if i%100 == 0:
			print(i)
	except ValueError:
		print('Error: ', filename)
		pass
