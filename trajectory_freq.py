import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def trajectory_freq(freq:float,interpEx, interpEy, interpEz, tot_time:float, dt:float, init_pos:np.array, init_vel:np.array, charges:np.array, mases:np.array, xmin:float, xmax:float, ymin:float, ymax:float, zmin:float, zmax:float):
    '''
    * freq: frequency of oscilation
    * interpEi: interpolator of each electric field component generated by
    scipy.interpolated.RegularGridInterpolator over a rectangular meshgrid with
    bounds xmin, xmax, ymin, ymax, zmin, zmax
    * tot_time: ellapsed time in which the trajectories will be calculated
    * dt: time step
    * init_pos: initial position np.array n x 3 (n particles)
    * charges: np.array n
    * mases: np.array n

    Retuns a 3D np.array of dimensions (number of time steps) x (number of particles) x
    x 3
    '''
    velocity=init_vel
    position=init_pos
    t=0
    Traj=[position]
    while t<tot_time:
        E_pos=np.array([interpEx(position), interpEy(position), interpEz(position)])
        E_pos = E_pos * np.cos(2 * np.pi * freq * t)
        E_pos = E_pos.T
        F_e = np.array([charges[i]*E_pos[i] for i in range(len(charges))])
        F_i=force_ions(position,charges)
        F=F_e+F_i
        a = np.array([F[i]/mases[i] for i in range(len(mases))])
        position = position + velocity*dt
        velocity = velocity + a*dt


        t=t+dt
        Traj = np.vstack ((Traj, [position]))

        '''potser no cal fer la seguent comprovacio, si la funcio dona error perque
        una particula se'n surt disminuim tot_time'''
        for pos in position:
            if pos[0] <= xmin or pos[0] >= xmax or pos[1] <= ymin or pos[1] >= ymax or pos[2] <= zmin or pos[2] >= zmax:
                print('ValueError: One of the requested xi is out of bounds')
                return Traj

    return Traj

def force_ions(positions: np.array, charges: np.array):
    k=9e9
    force=np.zeros((len(positions), 3))
    for j in range(len(positions)):
        for i in range(len(positions)):
            if i!=j:
                v_dist=np.subtract(positions[j],positions[i])
                mod=np.linalg.norm(v_dist)
                dist_3=mod**3
                scalar=(k*charges[j]*charges[i]/dist_3)
                force[j]=force[j]+scalar*v_dist
    return force


dt = 1e-4
tot_time = 2
num_ions=2
x0 = np.array([[-0.1,0.0,0.0],[0.1,0.0,0.0]]) #This should be an array of 3D arrays
v0 = np.array([[0,0,0],[0,0,0]])
#De moment, crec que millor condicions inicials simetriques
charg=np.ones(num_ions)*1.6e-19 #ion charge
m=np.ones(num_ions)*2.87347958e-25 #ion mass (chosen completely randomely)

y = np.linspace(-6, 6, 51)
z = np.linspace(-4, 4, 51)
x = np.linspace(-6, 6, 51)
Y, X, Z = np.meshgrid(y, x, z) #we only generate the pontential this way because of mayavi
V = np.load('V.npy')
Ex, Ey, Ez=np.gradient(-V)
interpEx = rgi((x, y, z), Ex)
interpEy = rgi((x, y, z), Ey)
interpEz = rgi((x, y, z), Ez)
points = np.array([[0,0,0], [1,1,1]])
freq = 1e5
Traj = trajectory_freq(freq,interpEx, interpEy, interpEz,tot_time,dt,x0,v0,charg,m,-6,6,-6,6,-4,4)
#print(Traj)

#pixar animation studios
frames = len(Traj)
sim_time = 30*1e3 #miliseconds
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
particles = ax.scatter(Traj[0,:,0], Traj[0,:,1], Traj[0,:,2], c=np.array(['blue', 'red']))
def update(frame):
    # for each frame, update the data stored on each artist.

    # update the scatter plot:
    particles._offsets3d = (Traj[frame, :, 0], Traj[frame, :, 1], Traj[frame, :, 2])
    plt.draw()
    return particles

def init():
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-4, 4)
    return particles

ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=sim_time/frames, init_func=init, blit=False, repeat=False)
plt.show()


#Plot the trajectory
ax = plt.figure().add_subplot(projection='3d')
for i in range(len(Traj[0])):
    ax.plot(Traj[:,i,0], Traj[:,i,1], Traj[:,i,2], label='Ion ' + str(i) + ' Trajectory')
    ax.scatter(Traj[0,i,0], Traj[0,i,1], Traj[0,i,2], label='initial position '+str(i))
ax.legend()
plt.show()