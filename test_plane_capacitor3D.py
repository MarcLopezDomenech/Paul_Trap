from functions3D import *
from mayavi import mlab

if __name__ == '__main__':
    '''
    we are going to create a two-plane capacitor with planes parallel to the
    YZ axes
    '''
    l = 10 #plane side
    d = 0.1 #distance between planes
    x = 0 #position of the first plane
    #y limits of the plane
    y1 = -l/2
    y2 = l/2
    #z limits of the plane
    z1 = -l/2
    z2 = l/2
    '''the plane mesh is just grid with lines parallel to the y and z axes
    Ny is the number of divisions of the y axis and Nz the divisions of the z axis'''
    Ny = 11
    Nz = 11
    obj1 = initial_plane_mesh(x, y1, y2, z1, z2, Ny, Nz)
    plot_mesh(obj1, 'Plane x = 0')

    x = d #position of the second plane
    obj2 = initial_plane_mesh(x, y1, y2, z1, z2, Ny, Nz)
    plot_mesh(obj2, 'Plane x = %f' %(d))

    obj = concatenate_meshes(obj1, obj2) #merge geoemtries of both planes into one object
    plot_mesh(obj, 'Both planes')

    V0 = 1 #potential difference between plates
    Vs = np.array([V0, 0])
    V = pot_surf(obj, Vs)
    #the first plane (x = 0) set to V0 and the second plane to null potential

    q = linear_system(obj, V) #array with the charge densities in each triangle

    C = 0.5 * l / (Ny - 1) * l / (Nz - 1) * sum(q[:len(obj1.bari)]) / V0

    print('Capacitance:', C)
    C_teor = l**2 / d
    print('Relative error = %2.6f %%' %(100*abs(C_teor-C)/C_teor))

    #PLOTS
    y = np.linspace(-l, l, 51)
    z = np.linspace(-l, l, 51)
    x = np.linspace(-d, 2*d, 51)
    Y, X, Z = np.meshgrid(y, x, z)
    #this is the correct meshgrid creation to avoid errors with mayavi

    print('Computing potential in all space...')
    V = potential2(obj, q, X, Y, Z)
    print('Potential calculation completed')

    X *= 200/3 #rescale x axis. mayavi doe not have option to rescale axes

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