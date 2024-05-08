import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from functions3D import obj, plot_mesh
import math as m
import pyvista as pv

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

def hiper(radius: float, center: np.array,num_points: int,circum:int):
    index=radius
    dist =float(radius) / float(circum)
    points=[]
    num = 0
    for i in range(circum):
        num=num+2*m.pi*(i+1)*dist
    dens=float(num_points/num)
    while index > 0:
        numero=int(dens*2*m.pi*index)
        for j in range(numero):
             x=index*m.cos(float(2*m.pi*j)/float(numero))
             y=index*m.sin(float(2*m.pi*j)/float(numero))
             z=m.sqrt(m.pow(x,2)+m.pow(y,2) +1)
             points.append(center+np.array([x,y,z]))
        index=index-dist
        points.append([0,0,1])
    return points

object = one_sheet_hyperboloid(0.1, 0.5, 1)
plot_mesh(object, 'One-sheet hyperboloid')

points=hiper(10.0,np.array([0,0,0]),5000,100)
X, Y, Z = zip(*points)
ax = plt.axes(projection ='3d')
ax.scatter(X, Y, Z, 'green')
plt.show()

cloud = pv.PolyData(points)
surf = cloud.delaunay_2d()
surf.plot(show_edges=True)