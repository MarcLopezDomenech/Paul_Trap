import numpy as np
import matplotlib.pyplot as plt
from surf_integral import *

epsilon0 = 8.854187812813e-12 #vaccum electric permittivity
epsilonr = 1 #relative permittivity
epsilon = epsilon0 * epsilonr #permittivity

def linear_system(mesh: object, V: np.array) -> np.array:
    '''
    mesh: object qith the mesh of all the surfaces
    V: np.array of size N, tells the potential in each mesh point

    Returns an array of size N with the surface charge density in each mesh point
    '''
    M=len(mesh.bari)
    N = len(mesh.vertex)
    Z = np.zeros((N, M))
    for i in range(M - 1):
        vert=mesh.topol(i)
        v1=mesh.vertex(vert(0))
        v2=mesh.vertex(vert(1))
        v3=mesh.vertex(vert(2))
        Z[:,i]=vec_func(mesh.vertex,v1, v2, v3)
    b = V
    q = np.linalg.solve(Z, b)
    return q

def pot_surf(mesh: object,V0: np.array) -> np.array:
    '''
    mesh: object qith the mesh of all the surfaces
    V0: np.array, with the potentials at each surface

    Returns an array of size M (number of baricenters) with the potential at each baricenter
    '''
    surf=mesh.ntriang
    V=np.zeros(sum(surf))
    M=sum(surf)
    V[:surf(0)]=V0(0)
    for i in range(1,M-1):
        V[surf(i-1):surf(i)]=V0(i)
    return V

def potential(mesh: object, q: np.array, pts: np.array) -> np.array:
    V=mat_func(pts,mesh.vertex,mesh.topol,q)
    V=V/4*pi*epsilon
    return V