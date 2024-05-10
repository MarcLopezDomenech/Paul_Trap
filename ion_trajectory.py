import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interpn
from functions3D import pot_surf, linear_system, potential2
from electrodes_geometry import object

def trajectory(X: np.array, Y: np.array, Z: np.array, V: np.array, tot_time:float, dt:float, init_pos:np.array, charges:np.array, mases:np.array):
    velocity=np.zeros((len(init_pos), 3))
    position=init_pos
    t=0
    E_vol=np.gradient(-V)
    Traj=[position]
    points=np.transpose(np.array([X,Y,Z]).reshape(3,-1))
    while t<tot_time:
        E_pos=interpn(points,E_vol,position)
        F_e = charges*E_pos
        F_i=force_ions(position,charges)
        F=F_e+F_i
        a = F/mases
        velocity = velocity + a*dt
        position = position + velocity*dt

        t=t+dt
        Traj = np.vstack ((Traj, position) )

    return Traj

def force_ions(positions: np.array, charges: np.array):
    k=9*m.pow(10,9)
    force=np.zeros((len(positions), 3))
    for j in range(len(positions)):
        for i in range(len(positions)):
            if i!=j:
                v_dist=np.subtract(positions[j],positions[i])
                mod=m.dist([0,0,0],v_dist)
                dist_3=m.pow(mod,3)
                scalar=(k*charges[j]*charges[i]/dist_3)
                force[j]=force[j]+scalar*v_dist
    return force


dt = 10e-3
num_ions=2
x0 = [[0,0,0],[0.1,0.1,0.1]] #This should be an array of 3D arrays
charg=np.ones((num_ions, 3))*1.6e-9 #ion charge
m=np.ones((num_ions, 3))*10e-25 #ion mass

V0 = 1
Vs = np.array([V0, -V0, -V0])
V = pot_surf(object, Vs)
q = linear_system(object, V)

#PLOTS
y = np.linspace(-6, 6, 51)
z = np.linspace(-4, 4, 51)
x = np.linspace(-6, 6, 51)
Y, X, Z = np.meshgrid(y, x, z)
V = potential2(object, q, X, Y, Z)
print(trajectory(X,Y,Z,V,1,dt,x0,charg,m))


