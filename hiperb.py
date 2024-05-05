import numpy as np
import math as m
import matplotlib.pyplot as plt

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
            points.append(center+np.array([index*m.cos(float(2*m.pi*j)/float(numero)),index*m.sin(float(2*m.pi*j)/float(numero)),1/m.cosh(index)]))
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
        numero=int(dens*2*m.pi*abs(index))
        for j in range(numero):
            points.append(center+np.array([rad*m.cos(float(2*m.pi*j)/float(numero)),rad*m.sin(float(2*m.pi*j)/float(numero)),index]))
        index=index-dist
    return points
    
points=hiper(10.0,np.array([0,0,0]),1000,20)
x, y, z = zip(*points)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(x, y, z, 'green')
plt.show()

points=lateral(20.0,np.array([0,0,0]),7000,10,20)
x, y, z = zip(*points)
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(x, y, z, 'green')
plt.show()