import numpy
import math
import sys, os
import vtk
import vtktools
import pickle

'''
Extract the real time for every time step.
'''

path = '../data/Indoor/run/'
time_step = []
for i in range(0,2501):
	try:
		filename = path + 'LSBUIndoor_' + str(i) + '.pvtu'
		print(filename)
		mesh = vtktools.vtu(filename)
		time_step.append((mesh.GetScalarField('Time'))[0])
		print(len(time_step))
	except:
		pass
		
print(len(time_step))
pickle.dump(time_step, open('time_step', 'wb'))
		
	
