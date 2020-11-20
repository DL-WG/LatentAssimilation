#!/usr/bin/env python

from numpy import *
from math  import *
import sys, os
sys.path.append("/home/lmottet/fluidity-temp/python/")
import numpy as np
import vtk
import vtktools
#from termcolor #import colored
import matplotlib.pyplot as plt
import fluidity_tools
import datetime, time
import pickle

# # # ########################## # # #
# # # ######  FUNCTIONS   ###### # # #
# # # ########################## # # #
#-------------------------------------------------#
#-- Function to read data in a text file         -#
#-------------------------------------------------#
def ReadData(path,filename,extension):

    output = []

    sf = open(path+filename+extension, 'r')
    data = sf.readlines()

    for i in range(0,len(data)):
        x = str.split(data[i])
        y = [float(v) for v in x]
        output.append(y)
    return output

# # # ########################## # # #
# # # ######    MAIN      ###### # # #
# # # ########################## # # #
if __name__ == '__main__':
    tic = time.time()

    #--------------------------------#
    #-- User input variables       --#
    #--------------------------------#
    # Physical variables
    coeff = 5.5e5 # Coefficient to convert Fluidity Tracer value into ppm concentration

    # Geometry variables
    Pi = np.pi
    theta = 110.0
    lz_box1 = 9.0 #// Length in the z-direction

    dx_box = 20*lz_box1# // x Position
    dy_box = 6*lz_box1 #// y Position
    dz_box = 0.0 #// z Position
    lx_box1 = 48.0# // Length in the x-direction
    ly_box1 = 8.11 #// Length in the y-direction

    e_room  = 0.5 # // Thickness of the walls 50cm

    #// Centre of the box (needed if a rotation is wanted)
    x_center = dx_box + lx_box1
    y_center = dy_box + ly_box1
    z_center = 0.0
    print 'x_center', x_center, 'y_center', y_center

    #// Parameters to define the room
    x_room = 42.2 #// x position of the left hand side corner
    y_room = e_room#// y position of the left hand side corner
    z_room = 6.08 #//z position of the left hand side corner

    #// Dimensions of the room
    lx_room = 5.3 #// length of the room
    ly_room = 7.11 #// width of the room
    lz_room = 2.42 #// height of the room


    xmin = dx_box+x_room#- x_center
    xmax = dx_box+x_room+lx_room#- x_center
    ymin = dy_box+y_room #- y_center
    ymax = dy_box+y_room+ly_room #- y_center
    zmin = dz_box+z_room
    zmax = dz_box+z_room+lz_room

    print 'xmin::', xmin, ' xmax::', xmax
    print 'ymin::', ymin, ' ymax::', ymax
    print 'zmin::', zmin, ' zmax::', zmax

    #// For the structured grid we want
    Nx   = 180
    Ny   = 250
    z0   = (zmax+zmin)/2. # height from the ground

    # Vtu files
    path          = '/home/maddalena/data/Indoor/run/'
    extension     = '.pvtu'
    name_simu     = 'LSBUIndoor'
    vtu_start     = 0
    vtu_end       = 2501
    vtu_step      = 1

    # ---------------------------#
    # -- Coordinates Fluidity --#
    #---------------------------#
    coordinates = []

    stepx = (xmax-xmin)/(Nx+1.)
    stepy = (ymax-ymin)/(Ny+1.)

    print 'stepx::', stepx, '; stepy::', stepy
    for i in range(Nx):
        x = xmin + stepx/2. + stepx*i
        for j in range(Ny):
            y = ymin + stepy/2. + stepy*j

            x0 = x - x_center # Centred on the box centre
            y0 = y - y_center
            x1 =   x0 * np.cos(theta*Pi/180.) + y0 * np.sin(theta*Pi/180.)
            y1 = - x0 * np.sin(theta*Pi/180.) + y0 * np.cos(theta*Pi/180.)
            x1 = x1 + x_center # Centred on the box centre
            y1 = y1 + y_center

            coordinates.append([x1,y1,z0])

    # print coordinates

    # sys.exit('\n EXIT HERE')

    #---------------------------------------------------------------------
    # EXTRACT DATA
    #---------------------------------------------------------------------
    tic = time.time()

    fieldname = 'Tracer'

    Tracer = {}
    Time   = {}
    errors = []

    for vtuID in range(vtu_start,vtu_end,vtu_step):
        t_s = time.time()
        try:
            filename=path+name_simu+'_'+str(vtuID)+extension
            print '\n\n  '+str(filename)

            Tracer[vtuID] = []
            Time[vtuID]   = []

            #-----------------------#
            #-- Using vtk library --#
            #-----------------------#
            # Read file
            print '     Read file'
            if filename[-4:] == ".vtu":
                gridreader=vtk.vtkXMLUnstructuredGridReader()
            elif filename[-5:] == ".pvtu":
                gridreader=vtk.vtkXMLPUnstructuredGridReader()
            gridreader.SetFileName(filename)
            gridreader.Update()
            ugrid=gridreader.GetOutput()

            # Initialise locator
            print '     Initialise cell Locator'
            CellLocator = vtk.vtkCellLocator()
            CellLocator.SetDataSet(ugrid)
            CellLocator.Update()

            # Initialise probe
            points = vtk.vtkPoints()
            points.SetDataTypeToDouble()

            # Check Validity of points
            print '     Gathering  coordinates'
            NrbPoints = 0
            for j in range(len(coordinates)):
                NrbPoints += 1
                points.InsertNextPoint(coordinates[j][0], coordinates[j][1], coordinates[j][2])

            print '           Set points into data...'
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            probe = vtk.vtkProbeFilter()

            print '           Map data into probe...', 'VTK version ::', vtk.vtkVersion.GetVTKMajorVersion(),'.', vtk.vtkVersion.GetVTKMinorVersion()
            if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
                probe.SetInput(polydata)
                probe.SetSource(ugrid)
            else:
                probe.SetInputData(polydata)
                probe.SetSourceData(ugrid)

            probe.Update()

            print '           Check point Validity'
            valid_ids = probe.GetOutput().GetPointData().GetArray('vtkValidPointMask')
            validPoints = vtktools.arr([valid_ids.GetTuple1(i) for i in range(NrbPoints)])
            print '           ... ', len(validPoints)-np.sum(validPoints), 'invalid points'

            # Extract data
            print'     Extract Data'
            values = probe.GetOutput().GetPointData().GetArray(fieldname)

            for j in range(len(coordinates)):
                # If valid point, extract using probe,
                # Otherwise extract the cell:
                #    If no cell associated - then it is really a non-valid point outside the domain
                #    Otherwise: do the average over the cell values - this provide the tracer value.
                if validPoints[j] == 1:
                    tmp = values.GetValue(j)
                    if tmp < 0.: # This is because a negative values of concentration makes no sense
                        tmp = 0.0
                    Tracer[vtuID].append(tmp*coeff)
                else:
                    coord_tmp = np.array(points.GetPoint(j))
                    cellID =  CellLocator.FindCell(coord_tmp) # cell ID which contains the sensor
                    idlist=vtk.vtkIdList()
                    ugrid.GetCellPoints(cellID, idlist)
                    pointsID_to_cellID = np.array([idlist.GetId(k) for k in range(idlist.GetNumberOfIds())]) # give all the points asociated with this cell
                    if len(pointsID_to_cellID) == 0: # Non-valid points - We assign negative value - like that we know we are outside the domain
                        Tracer[vtuID].append(-1e20)
                    else:
                        tmp = 0
                        for pointID in pointsID_to_cellID:
                            tmp += ugrid.GetPointData().GetArray(fieldname).GetTuple(pointID)[0]
                        tmp = tmp/len(pointsID_to_cellID)
                        if tmp < 0.: # This is because a negative values of concentration makes no sense
                            tmp = 0.0
                        Tracer[vtuID].append(tmp*coeff)
                toc = time.time()

            pickle.dump(Tracer[vtuID], open('Structured/'+str(vtuID),'wb'))
            t_e = time.time()
            print '\n\nTime : ', t_e - t_s, 'sec'

            # Time
            time_tmp = probe.GetOutput().GetPointData().GetArray('Time').GetValue(0)
            Time[vtuID].append(time_tmp)
        except:
            errors.append(vtuID)
            pass

    # print colored('\n Time in (s) ::', 'red', attrs=['bold'])
    # print Time
    # print colored('\n Tracer ::', 'red', attrs=['bold'])
    # print Tracer

    #-----------------------#
    #-- PLTOS             --#
    #-----------------------#
    for vtuID in range(vtu_start,vtu_end,vtu_step):
        try:
            plt.figure(1,figsize=(8, 6), dpi=2000)
            ax = plt.subplot(111)

            #--------------------------#
            #- Color map option       -#
            #--------------------------#
            my_cmap = plt.cm.get_cmap('gist_ncar') #'hot', 'RdYlGn', 'RdYlGn_r'
            my_cmap.set_under('grey') # This will put value under the minimum value specific in grey, i.e. our invalid points

            #--------------------------#
            #- Fluidity results       -#
            #--------------------------#
            Slice = np.reshape(Tracer[vtuID],(Nx,Ny))
            plt.imshow(Slice, cmap=my_cmap, interpolation='none', vmin=300.0, aspect=stepx/stepy)# interpolation = 'nearest'; cmap='hot'

            #--------------------------#
            #- Plot titles and legend -#
            #--------------------------#
            #legend
            cbar = plt.colorbar()
            cbar.set_label('Concentration ($ppm$)', rotation=270, fontsize=18)
            # plt.colorbar()
            plt.title('Value of $c$ over a slice at '+str(z0-zmin)+' m in the room'+'\n'+'Grid size:: '+str(Nx)+'x'+str(Ny))

            # plt.savefig('Tracer_'+str(vtuID)+'.svg')
            plt.savefig('Imgs_Structured/Tracer_'+str(vtuID)+'.png')
            plt.grid()
            # plt.show()
            plt.close()
        except:
            pass

    toc = time.time()

    print '\n\nTime : ', toc - tic, 'sec'
