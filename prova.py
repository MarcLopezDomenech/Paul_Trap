import numpy as np
import pyvista as pv

#triangular mesh d'un hiperboloide de dos cares
#codi adaptat de la web:
#https://tutorial.pyvista.org/tutorial/02_mesh/solutions/d_create-tri-surface.html

def two_sheets_hyperboloid(ulim):
    n = 100
    u = np.linspace(-ulim, ulim, n)
    u_ = np.array([u for _ in range(n)])
    u = u_
    v = np.linspace(0, 2*np.pi, n)
    v_ = np.array([v for _ in range(n)])
    v = v_.T
    x = np.sinh(u)*np.cos(v)
    y = np.sinh(u)*np.sin(v)
    z = np.cosh(u)
    return x, y, z

X, Y, Z = two_sheets_hyperboloid(3)
points = np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]
cloud = pv.PolyData(points)
cloud.plot(point_size=7)
surf = cloud.delaunay_2d()
surf.plot(show_edges=True)
#crec que els punts de la mesh estan emmagatzemats en surf.points i surf.faces
#conté els vèrtexs que formen cada triangle
#surf és del tipus pyvista.PolyData:
#https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.polydata#pyvista.PolyData