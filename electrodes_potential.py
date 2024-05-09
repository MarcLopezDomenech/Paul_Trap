import numpy as np
from electrodes_geometry import object
from functions3D import pot_surf, linear_system, potential2
from mayavi import mlab
import time

V0 = 1
Vs = np.array([V0, -V0, -V0])
V = pot_surf(object, Vs)
q = linear_system(object, V)

#PLOTS
y = np.linspace(-6, 6, 51)
z = np.linspace(-4, 4, 51)
x = np.linspace(-6, 6, 51)
Y, X, Z = np.meshgrid(y, x, z)
#this is the correct meshgrid creation to avoid errors with mayavi

print('Computing potential in all space...')
start = time.time()
V = potential2(object, q, X, Y, Z)
print('Potential calculation completed')
print('It took', time.time()-start, 's')

fig1 = mlab.figure()
#the following plots are equivalent to Matlab's slice
xslice = mlab.volume_slice(X, Y, Z, V, plane_orientation='x_axes', colormap='coolwarm', slice_index = 25)
yslice = mlab.volume_slice(X, Y, Z, V, plane_orientation='y_axes', colormap='coolwarm', slice_index = 25)
zslice = mlab.volume_slice(X, Y ,Z, V, plane_orientation='z_axes', colormap='coolwarm', slice_index = 25)
colorbar = mlab.colorbar(xslice, orientation='vertical') #add colorbar

[Ex, Ey, Ez] = np.gradient(-V)

#downsample not to have thousands of arrows / cones in the plot
Ex = Ex[::10, ::10, ::10]
Ey = Ey[::10, ::10, ::10]
Ez = Ez[::10, ::10, ::10]
Xd = X[::10, ::10, ::10]
Yd = Y[::10, ::10, ::10]
Zd = Z[::10, ::10, ::10]

#equivalent to Matlab's coneplot
coneplot = mlab.quiver3d(Xd, Yd, Zd, Ex, Ey, Ez, colormap='coolwarm', mode='cone')
mlab.title('Potential')
axes = mlab.axes()
mlab.show()

fig2 = mlab.figure()
contours = mlab.contour3d(X, Y, Z, V, colormap='coolwarm')
coneplot = mlab.quiver3d(Xd, Yd, Zd, Ex, Ey, Ez, colormap='coolwarm', mode='cone')
mlab.title('Potential')
axes = mlab.axes()
mlab.show()

#Matlab's contourslice
#'''
#Mayavi's and Matlab's meshgrids are different, that is why I have to
#recalculate everything
X,Y,Z = np.meshgrid(x,y,z)
V= potential2(object, q, X, Y, Z)
[Ex, Ey, Ez] = np.gradient(-V)
import matlab.engine
eng = matlab.engine.start_matlab()
eng.contourslice(X, Y, Z, V, np.linspace(-5, 5, 10), np.array([]), np.array([]))
eng.view(3)
#eng.eval('grid on', nargout=0)
eng.eval('colormap jet', nargout=0)
eng.eval('colorbar', nargout=0)
# Pause the script
input("Press Enter to exit Matlab plot...")
eng.exit()
#'''