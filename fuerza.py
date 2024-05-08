import numpy as np
import math as m
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

force_ions([[0,0,0],[0,0,1]],[1,1])
