import numpy as np
import matplotlib.pyplot as plt

def create_plane_mesh(N: int, p1: np.array, p2: np.array) -> tuple[np.array, float]:
    '''
    N: number of points of the mesh (N >= 2)
    p1: one extreme of the plane
    p2: other extrem of the plane
    p1 and p2 are arrays of dim 2

    Returns an array of dim N x 2 with the N points of the mesh and
    the distance between mesh points
    '''
    mesh = np.zeros((N, 2))
    for i in range(N):
        mesh[i, :] = i / (N -1) * p2[:] + (1 - i / (N -1)) * p1[:]
    length = np.linalg.norm(p2-p1)
    return mesh, length / (N - 1)

def plot_plane_mesh(p1: np.array, p2: np.array, mesh: np.array) -> None:
    '''
    p1: one extreme of the plane
    p2: other extrem of the plane
    p1 and p2 are arrays of dim 2
    mesh: mesh points, dim N x 2

    Plots the plane and the mesh points
    '''
    fig = plt.figure()
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
    plt.plot(mesh[:, 0], mesh[:, 1], 'o')
    plt.show()


def create_cilinder_mesh(N: int, c: np.array, r: float) -> tuple[np.array, float]:
    '''
    N: number of points of the mesh
    c: cilinder centre, array of dim 2
    r: cilinder radius

    Returns an array N x 2 with the mesh points and the distance between mesh points
    '''
    mesh = np.zeros((N, 2))
    for i in range(N):
        mesh[i, :] = c
        mesh[i] += np.array([np.cos(i*2*np.pi/N), np.sin(i*2*np.pi/N)])
    #h is the distance between mesh points. Calculated with law of cosines
    h = np.sqrt(2*r**2*(1-np.cos(2*np.pi/N)))
    return mesh, h

def plot_cilinder_mesh(c: np.array, r: float, mesh: np.array) -> None:
    '''
    c: cilinder centre
    r: cilinder radius
    mesh: mesh points

    Plots the cilinder and the mesh points
    '''
    fig = plt.figure()
    angles = np.linspace(0, 2*np.pi, 1001)
    x = c[0] + r*np.cos(angles)
    y = c[1] + r*np.sin(angles)
    plt.plot(x, y, '-')
    plt.plot(mesh[:, 0], mesh[:, 1], 'o')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    #test of plane geometry
    '''
    N = 10
    p1 = np.array([4, -5])
    p2 = np.array([0, 9])
    mesh, h = create_plane_mesh(N, p1, p2)
    plot_plane_mesh(p1, p2, mesh)
    '''
    
    #test of cilindrical geometry
    N = 50
    c = np.array([0,0])
    r = 1
    mesh, h = create_cilinder_mesh(N, c, r)
    plot_cilinder_mesh(c, r, mesh)