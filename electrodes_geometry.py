import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from functions3D import obj, plot_mesh

def one_sheet_hyperboloid(deltaz, zlim, r):
    z = np.linspace(-zlim, zlim, int(1/deltaz))
    deltaphi = deltaz / r
    phi = np.linspace(0, 2*np.pi, int(2*np.pi/deltaphi))
    Z, Phi = np.meshgrid(z, phi)
    points = np.c_[Z.reshape(-1), Phi.reshape(-1)]
    tri = Delaunay(points)
    '''
    fig = plt.figure()
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()
    '''
    points_3d = []
    for point in tri.points:
        points_3d.append([np.sqrt(1+point[0]**2)*np.cos(point[1]), np.sqrt(1+point[0]**2)*np.sin(point[1]), point[0]])
    points_3d = np.array(points_3d)
    bari = []
    for t in tri.simplices:
        bari.append(1/3*(points_3d[t[0]]+points_3d[t[1]]+points_3d[t[2]]))
    bari = np.array(bari)
    ntriang = len(tri.simplices)
    object = obj(points_3d, tri.simplices, bari, ntriang)
    return object

object = one_sheet_hyperboloid(0.1, 0.5, 1)
plot_mesh(object, 'One-sheet hyperboloid')