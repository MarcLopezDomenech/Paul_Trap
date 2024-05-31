import numpy as np
from functions2D import *
import matplotlib.pyplot as plt



if __name__ == '__main__':
    #Testing a two-cilinder capacitor

    #create first cilinder
    N = 200
    c1 = np.array([-1, 0]) #centre first cilinder
    r1 = 0.5 #radius first cilinder
    mesh1, h = create_cilinder_mesh(N, c1, r1)
    plot_cilinder_mesh(c1, r1, mesh1)

    #create second cilinder
    c2 = np.array([1, 0]) #centre second cilinder
    r2 = 0.5 #radius second cilinder
    mesh2, h = create_cilinder_mesh(N, c2, r2)
    plot_cilinder_mesh(c2, r2, mesh2)

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

    C_teo = np.pi*epsilon/np.arccosh(np.linalg.norm(c1)/r1) #Only works if c1=c2 and r1=r2
    print('Theoretical Capacitance (SI):', C_teo)
    E = abs(C-C_teo)
    r = E/C_teo
    print('Absolute Error (SI):', E)
    print('Relative Error:', r)

    #represent the potential
    # Generate x and y values
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
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
    plt.axis('equal')
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


    #We redo the calculations for 2N, to check how error changes
    mesh1, h = create_cilinder_mesh(2*N, c1, r1)

    #create second cilinder
    mesh2, h = create_cilinder_mesh(2*N, c2, r2)

    #merge both meshes in one
    mesh = np.zeros((2*2*N, 2))
    mesh[:2*N, :] = mesh1
    mesh[2*N:, :] = mesh2

    V0 = 1
    V = np.zeros(2*2*N)
    V[:2*N] = V0 / 2
    V[2*N:] = -V0 / 2
    q = linear_system(mesh, h, V)

    C = h * sum(q[:2*N]) / V0
    print('Capacitance for 2N (SI):', C)

    E2 = abs(C-C_teo)
    r2 = E2/C_teo
    print('Absolute Error for 2N(SI):', E2)
    print('Relative Error for 2N:', r2)
    print('Quocient errors absoluts:', E/E2)
    print('Quocient errors relatius:', r/r2)

    
    #Study of the variation of the error with N
    E_all=[]
    QE_all=[]
    r_all=[]
    Qr_all=[]
    N_all=[]
    for N in range(100, 400, 20):
        N_all.append(N)
        c1 = np.array([-1, 0]) #centre first cilinder
        r1 = 0.5 #radius first cilinder
        mesh1, h = create_cilinder_mesh(N, c1, r1)

        #create second cilinder
        c2 = np.array([1, 0]) #centre second cilinder
        r2 = 0.5 #radius second cilinder
        mesh2, h = create_cilinder_mesh(N, c2, r2)

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

        C_teo = np.pi*epsilon/np.arccosh(np.linalg.norm(c1)/r1) #Only works if c1=c2 and r1=r2
        E = abs(C-C_teo)
        r = E/C_teo
        E_all.append(E)
        r_all.append(r)

        #We redo for 2N
        mesh1, h = create_cilinder_mesh(2*N, c1, r1)

        #create second cilinder
        mesh2, h = create_cilinder_mesh(2*N, c2, r2)

        #merge both meshes in one
        mesh = np.zeros((2*2*N, 2))
        mesh[:2*N, :] = mesh1
        mesh[2*N:, :] = mesh2

        V0 = 1
        V = np.zeros(2*2*N)
        V[:2*N] = V0 / 2
        V[2*N:] = -V0 / 2
        q = linear_system(mesh, h, V)

        C = h * sum(q[:2*N]) / V0
        E2 = abs(C-C_teo)
        r2 = E2/C_teo
        QE_all.append(E/E2)
        Qr_all.append(r/r2)

    #plt.plot(np.log(N_all), np.log(E_all), '.--', label='Error absolut')
    #plt.legend()
    #plt.title('Error absolut en funció del nombre de punts N (escala logarítmica)')
    #plt.show()
    plt.plot(np.log(N_all), np.log(r_all), '.--', label='Error relatiu')
    #plt.legend()
    #plt.title('Error relatiu en funció del nombre de punts N (escala logarítmica)')
    plt.xlabel('Number of points N')
    plt.ylabel('Relative error (logarithmic scale)')
    plt.show()

    #plt.plot(N_all, QE_all, '.--', label='Quocient errors absoluts')
    plt.plot(N_all, Qr_all, '.--', label='Quocient errors relatius')
    #plt.legend()
    plt.title('Quotient of error for N points with error for 2N punts')
    plt.xlabel('Number of points N')
    plt.show()
        


