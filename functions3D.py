import numpy as np
import matplotlib.pyplot as plt
from surf_integral import *

epsilon0 = 1 #vaccum electric permittivity jeje
epsilonr = 1 #relative permittivity
epsilon = epsilon0 * epsilonr #permittivity

class obj():
        '''class to save an object's geometry. the object's mesh is made up of 
        triangles.
        vertex -> vertices of the triangles. np.array n x 3
        topol -> saves the vertices that constitute each triangle. np.array n x 3
        bari -> barycentres of the triangles. np.array n x 3
        ntriang -> when the geometry is actually made up of many object, ntriang
        is an array that contains the number of triangles of each object.
        one-dimensional np.array
        '''
        def __init__(self, vertex, topol, bari, ntriang):
            self.vertex = vertex
            self.topol = topol
            self.bari = bari
            self.ntriang = ntriang

def initial_plane_mesh(x, y1, y2, z1, z2, Ny, Nz):
    '''
    creates the geometry of plane of equation x = x. The triangles vertices
    form a rectangular grid
    y1, y2: plane limits in y axis
    z1, z2: plane limits in z axis
    Ny: number of grid divisions in the y axis
    Nz: number of grid divisions in the z axis
    '''
    y = np.linspace(y1, y2, Ny)
    z = np.linspace(z1, z2, Nz)
    vertex = []
    for i in range(Ny):
        for j in range(Nz):
            vertex.append(np.array([x, y[i], z[j]]))

    topol = []
    for i in range(Ny - 1):
        for j in range(Nz - 1):
            aux = i * Ny + j
            topol.append(np.array([aux, aux + 1, aux + Ny]))
            topol.append(np.array([aux + Ny, aux + 1, aux + Ny + 1]))

    bari = []

    for t in topol:
        bari.append(1/3*(vertex[t[0]] + vertex[t[1]] + vertex[t[2]]))
    
    ntriang = [len(topol)]

    res = obj(np.array(vertex), np.array(topol, dtype=int), np.array(bari), np.array(ntriang, dtype=int))
    return res

def plot_mesh(obj, title):
    '''plots the geometry saved in obj
    title: name of the plot'''
    # Creating figure
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(obj.vertex[:, 0], obj.vertex[:, 1], obj.vertex[:, 2], color = "green")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    #ax.plot([-0.1, -0.1], [-1, 1], [-1, 1])
    for t in obj.topol:
        #print(t)
        #print(obj.vertex[t[0]], obj.vertex[t[1]])
        ax.plot([obj.vertex[t[0]][0], obj.vertex[t[1]][0]], [obj.vertex[t[0]][1], obj.vertex[t[1]][1]], [obj.vertex[t[0]][2], obj.vertex[t[1]][2]], '-')
        ax.plot([obj.vertex[t[1]][0], obj.vertex[t[2]][0]], [obj.vertex[t[1]][1], obj.vertex[t[2]][1]], [obj.vertex[t[1]][2], obj.vertex[t[2]][2]], '-')
        ax.plot([obj.vertex[t[2]][0], obj.vertex[t[0]][0]], [obj.vertex[t[2]][1], obj.vertex[t[0]][1]], [obj.vertex[t[2]][2], obj.vertex[t[0]][2]], '-')

    ax.scatter3D(obj.bari[:, 0], obj.bari[:, 1], obj.bari[:, 2], color="black")

    plt.title(title)
    # show plot
    plt.show()

def concatenate_meshes(obj1, obj2):
    '''merges two geometries into a single one'''
    vertex = np.zeros((len(obj1.vertex) + len(obj2.vertex), 3))
    vertex[:len(obj1.vertex), :] = obj1.vertex
    vertex[len(obj1.vertex):, :] = obj2.vertex

    topol = np.zeros((len(obj1.topol) + len(obj2.topol), 3), dtype=int)
    topol[:len(obj1.topol), :] = obj1.topol
    topol[len(obj1.topol):, :] = obj2.topol + len(obj1.vertex)

    bari = np.zeros((len(obj1.bari) + len(obj2.bari), 3))
    bari[:len(obj1.bari), :] = obj1.bari
    bari[len(obj1.bari):, :] = obj2.bari

    ntriang = np.zeros(2, dtype=int)
    ntriang[0] = obj1.ntriang[0]
    ntriang[1] = obj2.ntriang[0]

    res = obj(vertex, topol, bari, ntriang)

    return res

def linear_system(obj: object, V: np.array) -> np.array:
    '''
    mesh: object with the mesh of all the surfaces
    V: np.array of size N, tells the potential in each mesh point

    Returns an array of size N with the surface charge density in each mesh point
    '''
    M=len(obj.bari)
    Z = np.zeros((M, M))
    for i in range(M):
        vert=obj.topol[i]
        v1=obj.vertex[vert[0]]
        v2=obj.vertex[vert[1]]
        v3=obj.vertex[vert[2]]
        guvec_func(obj.bari,v1, v2, v3, 1, Z[:, i])
    Z=Z/epsilon #omit 4*pi because it is already implemented in the guvec_func function
    q = np.linalg.solve(Z, V)
    return q

def pot_surf(obj: object,V0: np.array) -> np.array:
    '''
    mesh: object with the mesh of all the surfaces
    V0: np.array, with the potentials at each surface

    Returns an array of size M (number of baricenters) with the potential at each baricenter
    '''
    surf=obj.ntriang
    M=sum(surf)
    V=np.zeros(M)
    V[:surf[0]]=V0[0]
    a = 0
    b = surf[0]
    for i in range(1, len(surf)):
        a += surf[i-1]
        b += surf[i]
        V[a:b]=V0[i]
    return V

def potential(obj: object, q: np.array, pts: np.array) -> np.array:
    '''
    obj: geometry
    q: np.array dimension n, contains the charge densities
    pts: np.array n x 3. The points where we want to calculate the potential

    Returns an np.array of dimension n with the potential in each point
    '''
    V=np.zeros(len(pts))
    gumat_func(pts,obj.vertex,obj.topol,q, V)
    V=V/epsilon #omit 4*pi because it is already implemented in the gumat_func function
    return V

def potential2(obj: object, q: np.array, X: np.array, Y: np.array, Z: np.array) -> np.array:
    '''
    obj: geometry
    q: np.array dimension n, contains the charge densities
    X, Y, Z: three-dimensional np.arrays created with np.meshgrid with the
    points where we want to calculate the potential

    Returns a three-dimensional np.array, same shape as X, Y and Z with the
    potential in each point of the grid
    '''
    pts = []
    a, b, c = np.shape(X)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                pts.append([X[i,j,k], Y[i,j,k], Z[i,j,k]])
    V_ = np.zeros(len(pts))
    gumat_func(pts, obj.vertex, obj.topol, q, V_)
    V = np.zeros((a, b, c))
    l = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                V[i, j, k] = V_[l]
                l += 1
    return V
