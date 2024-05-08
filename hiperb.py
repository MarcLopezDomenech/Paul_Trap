import numpy as np
import math as m
import matplotlib.pyplot as plt
import pyvista as pv

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
    return points

def lateral(radius: float, center: np.array,num_points: int,altura:float, circum:int):
    dist =float(altura) / float(circum)
    index=altura/2
    points=[]
    num = 0
    for i in range(circum):
        num=num+2*m.pi*radius*m.cosh(m.dist([radius,0,(i+1)*dist-altura/2], [0,0,0])-radius)
    dens=float(num_points/num)
    while index > -altura/2:
        rad=radius*m.cosh(m.dist([radius,0,index], [0,0,0])-radius)
        numero=int(dens*2*m.pi*abs(rad))
        for j in range(numero):
            points.append(center+np.array([rad*m.cos(float(2*m.pi*j)/float(numero)),rad*m.sin(float(2*m.pi*j)/float(numero)),index]))
        index=index-dist
    return points
    
points=hiper(10.0,np.array([0,0,0]),1000,100)
x, y, z = zip(*points)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(x, y, z, 'green')
plt.show()
cloud = pv.PolyData(points)
surf = cloud.delaunay_2d()
surf.plot(show_edges=True)
surf.plot(cpos="yz", show_edges=True)

points=lateral(20.0,np.array([0,0,0]),10000,10,20)
x, y, z = zip(*points)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.scatter(x, y, z, 'green')
plt.show()
cloud = pv.PolyData(points)
surf = cloud.delaunay_2d()
surf.plot(show_edges=True)
surf.plot(cpos="yz", show_edges=True)