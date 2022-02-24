#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: comecattin
"""
import numpy as np
import matplotlib.pyplot as plt
import MPCMolecularDynamics as MD
from tqdm import tqdm


#Define the system
d=2
NParticles=16
m = np.ones(NParticles)


# Lennard-Jones
sigma = 1
epsilon = 1
tau = sigma*np.sqrt(m[0]/epsilon) #unit of time    

# define the length of the trajectory
run_time = 100. * tau


kT = np.linspace(0.3,1,5)
rho  = np.linspace(0.05,0.8,5)/sigma**d

LBox = np.power(NParticles/rho,1./d)

compressibility = np.zeros((kT.shape[0],LBox.shape[0]))

for i, LBox_i in enumerate(LBox):
    
    for j, kT_j in enumerate(kT):
        
        ( t_tr, x_tr, y_tr, 
         vx_tr, vy_tr, 
         uPot_tr, uKin_tr, 
         pPot_tr, pKin_tr, 
         pHyper_tr, 
         [t,x,y,vx,vy]) = MD.Generate_LJ_NVT_MolecularDynamics_Trajectory(d,m,
                                                                          LBox_i,
                                                                          kT_j,
                                                                          run_time,
                                                                          number_of_time_steps_between_stored_configurations=1)

        compressibility[i,j] = MD.Compressibility_from_pressure_fluctuations_in_NVT(d,m,
                                                                                    NParticles,
                                                                                    LBox_i,
                                                                                    kT_j,
                                                                                    pPot_tr, 
                                                                                    pHyper_tr, 
                                                                                    pKin_tr)

#Plot
plt.imshow(compressibility)
plt.savefig('phase_diagramm.png')
plt.xlabel("kT")
plt.ylabel("Rho")
