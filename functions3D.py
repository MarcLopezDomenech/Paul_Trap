import numpy as np
import matplotlib.pyplot as plt
from surf_integral import *

epsilon0 = 8.854187812813e-12 #vaccum electric permittivity
epsilonr = 1 #relative permittivity
epsilon = epsilon0 * epsilonr #permittivity

class obj():
        def __init__(self, vertex, topol, bari, ntriang):
            self.vertex = vertex
            self.topol = topol
            self.bari = bari
            self.ntriang = ntriang

def initial_plane_mesh(x, y1, y2, z1, z2, Ny, Nz):
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

    res = obj(np.array(vertex), np.array(topol, dtype=int), np.array(bari), np.array(ntriang))
    return res

def plot_mesh(obj):
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

    plt.title("Plane mesh")
    # show plot
    plt.show()

def concatenate_meshes(obj1, obj2):
    vertex = np.zeros((len(obj1.vertex) + len(obj2.vertex), 3))
    vertex[:len(obj1.vertex), :] = obj1.vertex
    vertex[len(obj1.vertex):, :] = obj2.vertex

    topol = np.zeros((len(obj1.topol) + len(obj2.topol), 3), dtype=int)
    topol[:len(obj1.topol), :] = obj1.topol
    topol[len(obj1.topol):, :] = obj2.topol + len(obj1.vertex)

    bari = np.zeros((len(obj1.bari) + len(obj2.bari), 3))
    bari[:len(obj1.bari), :] = obj1.bari
    bari[len(obj1.bari):, :] = obj2.bari

    ntriang = np.zeros(2)
    ntriang[0] = obj1.ntriang[0]
    ntriang[1] = obj2.ntriang[0]

    res = obj(vertex, topol, bari, ntriang)

    return res


if __name__ == '__main__':
    x = -0.1
    y1 = -1
    y2 = 1
    z1 = -1
    z2 = 1
    Ny = 3
    Nz = 3
    obj1 = initial_plane_mesh(x, y1, y2, z1, z2, Ny, Nz)
    '''
    print('vertex =', obj1.vertex)
    print('topol=', obj1.topol)
    print('bari=', obj1.bari)
    print('ntriang=', obj1.ntriang)
    '''
    plot_mesh(obj1)

    x = 0.1
    y1 = -1
    y2 = 1
    z1 = -1
    z2 = 1
    Ny = 4
    Nz = 4
    obj2 = initial_plane_mesh(x, y1, y2, z1, z2, Ny, Nz)
    #'''
    print('vertex =', obj2.vertex)
    print('topol=', obj2.topol)
    print('bari=', obj2.bari)
    print('ntriang=', obj2.ntriang)
    #'''
    plot_mesh(obj2)

    obj = concatenate_meshes(obj1, obj2)

    print('vertex =', obj.vertex)
    print('topol=', obj.topol)
    print('bari=', obj.bari)
    print('ntriang=', obj.ntriang)

    plot_mesh(obj)

def linear_system(obj: object, V: np.array) -> np.array:
    '''
    mesh: object with the mesh of all the surfaces
    V: np.array of size N, tells the potential in each mesh point

    Returns an array of size N with the surface charge density in each mesh point
    '''
    M=len(obj.bari)
    N = len(obj.vertex)
    Z = np.zeros((N, M))
    for i in range(M):
        vert=obj.topol[i]
        v1=obj.vertex(vert[0])
        v2=obj.vertex(vert[1])
        v3=obj.vertex(vert[2])
        Z[:,i]=vec_func(obj.bari,v1, v2, v3)
    Z=Z/4*pi*epsilon
    b = V
    q = np.linalg.solve(Z, b)
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
    for i in range(1,len(surf)):
        V[surf[i-1]:surf(i)]=V0[i]
    return V

def potential(obj: object, q: np.array, pts: np.array) -> np.array:
    V=mat_func(pts,obj.vertex,obj.topol,q)
    V=V/4*pi*epsilon
    return V