from functions3D import *

if __name__ == '__main__':
    l = 10 #plane side
    d = 0.1 #distance btwn planes
    x = 0
    y1 = -l/2
    y2 = l/2
    z1 = -l/2
    z2 = l/2
    Ny = 11
    Nz = 11
    obj1 = initial_plane_mesh(x, y1, y2, z1, z2, Ny, Nz)
    plot_mesh(obj1)

    x = d
    obj2 = initial_plane_mesh(x, y1, y2, z1, z2, Ny, Nz)
    plot_mesh(obj2)

    obj = concatenate_meshes(obj1, obj2)
    plot_mesh(obj)

    V0 = 1
    Vs = np.array([V0, 0])
    V = pot_surf(obj, Vs)

    q = linear_system(obj, V)

    C = 0.5 * l / (Ny - 1) * l / (Nz - 1) * sum(q[:len(obj1.bari)]) / V0

    print('Capacitance:', C)
    C_teor = l**2 / d
    print('Relative error = %2.6f %%' %(100*abs(C_teor-C)/C_teor))