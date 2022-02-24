#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: comecattin
"""

import numpy as np
import multiprocessing as mp
import itertools
import MPCMolecularDynamics as MD
import matplotlib.pyplot as plt



def Generate_LJ_NVT_MolecularDynamics_Trajectory(d,m,system,run_time,
                                                 starting_configuration=[],
                                                 time_step = 0.01,
                                                 number_of_time_steps_between_stored_configurations=100,
                                                 number_of_time_steps_between_velocity_resets=100,
                                                 start_from_stable_grid = False,
                                                 debug=False):
    """
    generates a NVT MD simulations of a LJ system with sigma=epsilon=1  
        - where the particle masses are specified in the array m
        - so that NParticles = m.size
        - in a volume V=(LBox,LBox) at a specified temperature kT
        - with a time step of time_step tau 
          where the LJ unit of time is calculated as a function of m[-1], i.e. the mass of the LAST particle
        - runs are either started from 
                a specified starting configuration [t,x,y,vx,vy] or
                initialized with zero velocities and particles placed on a square grid
        - the simulations are thermostated by redrawing random velocities from the 
          Maxwell-Boltzmann distribution number_of_time_steps_between_velocity_resets time steps
        - the function returns 
                trajectory lists t_tr, x_tr, y_tr, vx_tr, vy_tr, uPot_tr, uKin_tr, pPot_tr, pKin_tr, pHyper_tr 
                     of sampling times and sampled coordinates, velocities and energies and pressures
                a final configuration [t,x,y,vx,vy] from which the run can be restarted
                while the energies and pressures are recorded at every time step, configurations 
                     and velocities are stored at a time interval of time_between_stored_configurations
                    
    """
    #Added this line to make multiprossessing easier
    LBox, kT = system
    
    NParticles = m.size
    sigma = 1
    epsilon = 1
    #unit of time
    #tau = sigma*np.sqrt(m[-1]/epsilon)      

    # define the length of the trajectory
    number_of_timesteps = int(np.round(run_time/time_step))

    #starting configuration
    if starting_configuration!=[]:
        [t,x,y,vx,vy] = starting_configuration
    else:
        # default initial state
        if start_from_stable_grid:
            x,y = MD.StableGridPositionsIn2d(LBox,LBox,NParticles)
        else:
            x,y = MD.GridPositionsIn2d(LBox,LBox,NParticles)
        vx = MD.RandomVelocities(m,kT)
        vy = MD.RandomVelocities(m,kT)
        t = 0
        if debug:
            print("No starting configuration")

    #initialize Trajectory
    t_tr = []
    x_tr = []
    vx_tr = []
    y_tr = []
    vy_tr = []

    fx,fy = MD.LJ_forces_as_a_function_of_positions(d,epsilon,sigma,LBox,(x,y))
    # force for initial configuration needed for first time step

    for timestep in range(number_of_timesteps):
        (x,y),(vx,vy) = MD.VelocityVerletTimeStepPartOne(m,(x,y),(vx,vy),(fx,fy),time_step)
        fx,fy = MD.LJ_forces_as_a_function_of_positions(2,epsilon,sigma,LBox,(x,y))
        (x,y),(vx,vy) = MD.VelocityVerletTimeStepPartTwo(m,(x,y),(vx,vy),(fx,fy),time_step)
        t += time_step
        
        t_tr.append(t)
        x_tr.append(x)
        vx_tr.append(vx)
        y_tr.append(y)
        vy_tr.append(vy)
    
        # thermostat: reinitialise velocities to control temperature
#        if np.mod( timestep*time_step, time_between_velocity_resets ) == 0.0 and timestep>1:
        if timestep%number_of_time_steps_between_velocity_resets == 0 and timestep>1:
            vx = MD.RandomVelocities(m,kT)
            vy = MD.RandomVelocities(m,kT)

    # convert trajectory lists to arrays to simplify the data analysis
    t_tr = np.array(t_tr)
    x_tr = np.array(x_tr)
    vx_tr = np.array(vx_tr)
    y_tr = np.array(y_tr)
    vy_tr = np.array(vy_tr)

    # analyse results 
    uPot_tr = MD.LJ_energy_as_a_function_of_positions(d,epsilon,sigma,LBox,(x_tr,y_tr))
    uKin_tr = MD.TotalKineticEnergy(m,vx_tr) + MD.TotalKineticEnergy(m,vy_tr)
    pPot_tr = MD.LJ_virial_pressure_as_a_function_of_positions(d,epsilon,sigma,LBox,(x_tr,y_tr)) 
    pKin_tr = MD.KineticPressure_as_a_function_of_velocities(d,LBox,m,(vx_tr,vy_tr))
    pHyper_tr = MD.LJ_hyper_virial_as_a_function_of_positions(d,epsilon,sigma,LBox,(x_tr,y_tr)) 
    
    # reduce the number of stored configurations and velocities
#    skip = int(time_between_stored_configurations / delta_t)
    skip = number_of_time_steps_between_stored_configurations
    x_tr = x_tr[::skip]
    y_tr = y_tr[::skip]
    vx_tr = vx_tr[::skip]
    vy_tr = vy_tr[::skip]    
    # note that t_tr is not compressed as it contains the times corresponding to the stored energies and pressures
    # as a consequence a corresponding skipping operation needs to be performed, when configurations are plotted 
    # as a function of time
    
    return t_tr, x_tr, y_tr, vx_tr, vy_tr, uPot_tr, uKin_tr, pPot_tr, pKin_tr, pHyper_tr, [t,x,y,vx,vy]


def get_results(d,m,system,run_time):
    """
    The routine that will be executed in each iteration into multiprocessing.
    First generate the trajectory via MD LJ, then calculate the compressibility
    of the system.
    
    
    Parameters
    ----------
    d : int
        Dimension of the LJ system.
    m : array ; size = number of particles
        Array of the mass of each particles
    system: iterrable
         iterrable containnig LBox (length of the box, iterable) 
         and kT (temperature, iterable) of the system.
    run_time : int
        How long the simulation will be.
        
        
    Returns
    -------
    compressibility : Array
        Compressibility of the system for every couple of LBox and kT given in
        system.

    """
    
    #Extract data
    LBox, kT = system
    
    
    #Generate trajectories
    ( t_tr, x_tr, y_tr, 
     vx_tr, vy_tr, 
     uPot_tr, uKin_tr, 
     pPot_tr, pKin_tr, 
     pHyper_tr, 
     [t,x,y,vx,vy]) = Generate_LJ_NVT_MolecularDynamics_Trajectory(d,m,
                                                                      system,
                                                                      run_time,
                                                                      number_of_time_steps_between_stored_configurations=1)
    
    #Compute the compressibility
    compressibility = MD.Compressibility_from_pressure_fluctuations_in_NVT(d,m,
                                                                                NParticles,
                                                                                LBox,
                                                                                kT,
                                                                                pPot_tr, 
                                                                                pHyper_tr, 
                                                                                pKin_tr)
    
    return compressibility



def main(d,NParticles,sigma,epsilon,kT,rho):
    """
    

    Parameters
    ----------
    d : int
        Dimension of the LJ system.
    NParticles : int
        Number of particles in the system
    sigma : float
        Sigma parameter in the LJ system.
    epsilon : float
        Epsilon parameter in the LJ system.
    kT : iterable of float
        Temperatures at which the system is to be computed
    rho : iterable of float
        Density at which the system is to be computed

    Returns
    -------
    compressibility : Array
        Compressibility of the system for every couple of rho and kT

    """
    
    #the System
    m = np.ones(NParticles)
    LBox = np.power(NParticles/rho,1./d)
    
    #Unit of time
    tau = sigma*np.sqrt(m[0]/epsilon)
    
    # define the length of the trajectory
    run_time = 100. * tau
    
    
    #Multiprocessing
    pool = mp.Pool(mp.cpu_count())
    #Every possible couple of rho and kT
    couple = list(itertools.product(LBox,kT))

    #Compute the compressibility
    compressibility = pool.starmap(
        get_results,
        [(d, m, (LBox_i, kT_j), run_time) for LBox_i, kT_j in couple]
        )
    
    #End of multiprocessing
    pool.close()
    
    #Convert to np.array for better data analysis
    compressibility = np.array(compressibility)
    #Reshape to have a square matrix
    compressibility = compressibility.reshape((len(rho),len(kT)))
    
    
    
    #Plot
    plt.imshow(compressibility,extent=(0.05,0.8,0.3,5),aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Compressibility')
    plt.xlabel('rho')
    plt.ylabel('kT')
    plt.title('Phase diagramm')
    plt.savefig('phase_diagramm.png')
    
    return compressibility
    
if __name__ == '__main__':
    
    #Dimension
    d=2
    #Number of particles
    NParticles=16
    

    # Lennard-Jones
    sigma = 1
    epsilon = 1

    #Temperature and density
    kT = np.linspace(0.3,1,5)
    rho  = np.linspace(0.05,0.8,5)/sigma**d
    
    #Get compressibility
    compressibility = main(d,NParticles,sigma,epsilon,kT,rho)
    #Save it to a .npy file
    np.save('compressibility.npy',compressibility)










