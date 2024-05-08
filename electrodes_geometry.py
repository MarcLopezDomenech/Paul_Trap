import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from functions3D import obj, plot_mesh, concatenate_meshes
import math as m
import pyvista as pv
import sys

def one_sheet_hyperboloid(deltaz, zlim, r, a):
    z = np.linspace(-zlim, zlim, int(2*zlim/deltaz))
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
        points_3d.append([r*np.sqrt(1+a*point[0]**2)*np.cos(point[1]), r*np.sqrt(1+a*point[0]**2)*np.sin(point[1]), point[0]])
    points_3d = np.array(points_3d)
    bari = []
    for t in tri.simplices:
        bari.append(1/3*(points_3d[t[0]]+points_3d[t[1]]+points_3d[t[2]]))
    bari = np.array(bari)
    ntriang = np.array([len(tri.simplices)])
    object = obj(points_3d, tri.simplices, bari, ntriang)
    return object

def two_sheets_hyperboloid(radius: float, center: np.array,num_points: int,circum:int, a:float):
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
             z=m.sqrt(a*(m.pow(x,2)+m.pow(y,2)) +1)
             points.append(center+np.array([x,y,z]))
        index=index-dist
    points.append([0,0,1])

    X, Y, Z = zip(*points)
    #ax = plt.axes(projection ='3d')
    #ax.scatter(X, Y, Z, 'green')
    #plt.show()
    cloud = pv.PolyData(points)
    surf = cloud.delaunay_2d()
    #surf.plot(show_edges=True)
    topol = []
    t = []
    for i in range(1, len(surf.faces)):
        if i % 4 == 0:
            topol.append(t)
            t = []
        else:
            t.append(surf.faces[i])
    topol.append(t)
    topol = np.array(topol)
    bari = []
    for t in topol:
        bari.append(1/3*(np.array(surf.points[t[0]])+np.array(surf.points[t[1]])+np.array(surf.points[t[2]])))
    bari = np.array(bari)
    ntriang = np.array([len(topol)])

    object2 = obj(surf.points, topol, bari, ntriang)
    return object2

def two_sheets_hyperboloid2(radius: float, center: np.array,num_points: int,circum:int, a:float):
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
             z=-m.sqrt(a*(m.pow(x,2)+m.pow(y,2)) +1)
             points.append(center+np.array([x,y,z]))
        index=index-dist
    points.append([0,0,-1])

    X, Y, Z = zip(*points)
    #ax = plt.axes(projection ='3d')
    #ax.scatter(X, Y, Z, 'green')
    #plt.show()
    cloud = pv.PolyData(points)
    surf = cloud.delaunay_2d()
    #surf.plot(show_edges=True)
    topol = []
    t = []
    for i in range(1, len(surf.faces)):
        if i % 4 == 0:
            topol.append(t)
            t = []
        else:
            t.append(surf.faces[i])
    topol.append(t)
    topol = np.array(topol)
    bari = []
    for t in topol:
        bari.append(1/3*(np.array(surf.points[t[0]])+np.array(surf.points[t[1]])+np.array(surf.points[t[2]])))
    bari = np.array(bari)
    ntriang = np.array([len(topol)])

    object2 = obj(surf.points, topol, bari, ntriang)
    return object2

object1 = one_sheet_hyperboloid(0.4, 1, 5, 0.25)
#plot_mesh(object1, 'One-sheet hyperboloid')

object2 = two_sheets_hyperboloid(7.0, np.array([0, 0, 0]), 200, 10, 0.25)
#plot_mesh(object2, 'Two-sheets hyperboloid')

object3 = two_sheets_hyperboloid2(7.0, np.array([0, 0, 0]), 200, 10, 0.25)
#plot_mesh(object3, 'Two-sheets hyperboloid') 

object = concatenate_meshes(object1, object2)
object = concatenate_meshes(object, object3)
plot_mesh(object, 'Two electrodes together')