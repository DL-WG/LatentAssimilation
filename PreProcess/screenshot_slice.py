import sys, os
import pyvista as pv
import numpy as np
import pickle
from pathlib import Path

'''
Given the slice for each time step, compute a screenshot coloured by tracer values.
The scalar bar of the image considers the global minimum and the global maximum found in both datasets and sensor data.
'''

path = './Slice/'
dest_plot = './RGB_plot/'
rgbs = './RGB/'
Path(dest_plot).mkdir(parents=True, exist_ok=True)
Path(rgbs).mkdir(parents=True, exist_ok=True)

coeff = 5.5e5 #coefficient to convert the tracer value in PPM scale
ranges = pickle.load(open('range', 'rb'))
min_ = ranges['min']/coeff
max_ = ranges['max']/coeff
print('Range: ',min_, max_)

#files=os.listdir(rgbs)

for filename in os.listdir(path):
    if(not(os.path.isdir(filename)) and 'sLSBUIndoor' in filename):
        #if(filename.split('_')[1].split('.')[0] not in files):
           # print(filename)
        try:
            single_slice = pv.read(path+filename)
            single_slice.set_active_scalars('Tracer')
            plotter = pv.Plotter(off_screen=True)
            plotter.camera_set = True
            plotter.camera.SetPosition([225.31895446777344, 66.47050285339355, 18.83050280591159])
            plotter.camera.SetViewUp([0.34202014332566816, 0.9396926207859086, 0])
            plotter.camera.SetFocalPoint([225.31895446777344, 66.47050285339355, 7.289999961853027])
            plotter.camera.SetFocalDisk(1)
            plotter.camera.SetViewAngle(30)
            plotter.camera.SetEyeAngle(2)
            plotter.camera.Dolly(1)
            plotter.add_mesh(single_slice)
            plotter.update_scalar_bar_range(clim=[min_, max_])
            plotter.remove_scalar_bar()
            name =filename.split('_')[1].split('.')[0]
            img = plotter.screenshot(dest_plot + name, transparent_background = True, return_img = True, window_size=(260,215))
            pickle.dump(img, open( rgbs + name, 'wb'))
        except:
            print('Error: ', filename)
            pass
