import numpy as np
from functions2D import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #testing a plane capacitor

    #create first plane
    N = 200
    p1 = np.array([-0.1, 5])
    p2 = np.array([-0.1, -5])
    mesh1, h = create_plane_mesh(N, p1, p2)
    plot_plane_mesh(p1, p2, mesh1)

    p1 = np.array([0.1, 5])
    p2 = np.array([0.1, -5])
    mesh2, h = create_plane_mesh(N, p1, p2)
    plot_plane_mesh(p1, p2, mesh2)

    #merge both meshes in one
    mesh = np.zeros((2*N, 2))
    mesh[:N, :] = mesh1
    mesh[N:, :] = mesh2

    V0 = 1
    V = np.zeros(2*N)
    V[:N] = V0 / 2
    V[N:] = -V0 / 2
    q = linear_system(mesh, h, V)

    C = h * sum(q[:N]) / V0
    print('Capacitance (SI):', C)

    #represent the potential
    # Generate x and y values
    x = np.linspace(-0.15, 0.15, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = potential(mesh, h, q, X, Y)

    # Create a 3D surface plot
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('V')
    ax.set_title('Potential')

    # Add color bar
    fig1.colorbar(surf, ax=ax, label='')

    plt.show()

    # Create a contour plot
    fig2 = plt.figure()
    plt.contourf(X, Y, Z, cmap='coolwarm', levels=100)
    plt.colorbar(label='')
    plt.contour(X, Y, Z, colors='k', linestyles='solid', linewidths=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Potential and gradient')

    # Define vector field (example: gradient of a scalar function)
    [Ey, Ex] = np.gradient(-Z) #it's strange I have to interchange ex, Ey here
    X_downsampled = X[::5, ::5]
    Y_downsampled = Y[::5, ::5]
    Z_downsampled = Z[::5, ::5]
    Ex_downsampled = Ex[::5, ::5]
    Ey_downsampled = Ey[::5, ::5]
    # Plot vector field
    plt.quiver(X_downsampled, Y_downsampled, Ex_downsampled, Ey_downsampled, scale = 1)
    plt.show()