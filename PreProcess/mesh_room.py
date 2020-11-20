import numpy
import math
import sys, os
import vtk

'''
Build an unstructured mesh from the points of the room.
The vtu files will be stored in Mesh folder.
'''

path = './Room/'
for filename in os.listdir(path):

	reader=vtk.vtkXMLUnstructuredGridReader()
	reader.SetFileName(path+filename)
	reader.Update()
	ugrid_=reader.GetOutput()

	delny = vtk.vtkDelaunay3D()
	delny.SetInputConnection(reader.GetOutputPort())
	delny.SetTolerance(0.01)
	delny.SetAlpha(0.2)

	out_wrt = vtk.vtkXMLUnstructuredGridWriter()
	out_wrt.SetInputConnection(delny.GetOutputPort())
	out_wrt.SetFileName('Mesh/m'+filename)
	out_wrt.Write()