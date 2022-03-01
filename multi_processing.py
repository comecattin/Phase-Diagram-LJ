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
    
    MODIFIED FUNCTION FOR MULTIPROSSESSING
    
    
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
    print(time_step)
    
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


def Generate_Ensemble_of_LJ_NVT_MolecularDynamics_Trajectories(d,m,system,NTrajectories,run_time,
                                                               list_of_starting_configurations=[],
                                                               time_step=0.01,
                                                               number_of_time_steps_between_stored_configurations=100,
                                                               number_of_time_steps_between_velocity_resets=100,
                                                               start_from_stable_grid=False,
                                                               debug=False):
    """
    
    MODIFIED FUNCTION FOR MULTIPROSSESSING
    
    
    uses Generate_LJ_NVT_MolecularDynamics_Trajectory to

    generate an ensemble of NTrajectories NVT MD simulations of a LJ system with sigma=epsilon=1  
        - where the particle masses are specified in the array m
        - so that NParticles = m.size
        - in a volume V=(LBox,LBox) at a specified temperature kT
        - with a time step of time_step tau 
          where the LJ unit of time is calculated as a function of m[-1], i.e. the mass of the LAST particle
        - runs are either started from 
                a list of specified starting configuration [[t,x,y,vx,vy], ...] or
                initialized with zero velocities and particles placed on a square grid
        - the simulations are thermostated by redrawing random velocities from the 
          Maxwell-Boltzmann distribution at intervals of time_between_velocity_resets
        - the function returns 
                trajectory ensemble lists t_tr_ens, x_tr_ens, y_tr_ens, vx_tr_ens, vy_tr_ens, uPot_tr_ens, uKin_tr_ens, pPot_tr_ens, pKin_tr_ens, pHyper_tr_ens 
                     of sampling times and sampled coordinates, velocities and energies and pressures
                a list of final configurations [[t,x,y,vx,vy], ...] from which the runs can be restarted
                while the energies and pressures are recorded at every time step, configurations 
                     and velocities are stored at a time interval of time_between_stored_configurations      
    """
    
    
    #Added this line to make multiprossessing easier
    LBox, kT = system
    
    
    
    # initialize lists to collect ENSEMBLES of trajectories
    t_tr_ens = []
    x_tr_ens = []
    vx_tr_ens = []
    y_tr_ens = []
    vy_tr_ens = []
    uKin_tr_ens = []
    uPot_tr_ens = []
    pKin_tr_ens = []
    pPot_tr_ens = []
    pHyper_tr_ens = []
    
    # convert empty list into lists of NTrajectories empty lists, 
    # which can then by passed on to the simulation routine
    if list_of_starting_configurations==[]:
        local_list_of_starting_configurations=[]
        if debug:
            print("No list of starting configurations")
        for n in range(NTrajectories): 
            local_list_of_starting_configurations.append([])
    else:
        local_list_of_starting_configurations = list_of_starting_configurations

    for n in range(NTrajectories):
        if debug:
            print('.', end='', flush=True)
        (t_tr, x_tr, y_tr, vx_tr, vy_tr, 
         uPot_tr, uKin_tr, pPot_tr, pKin_tr, pHyper_tr, 
         local_list_of_starting_configurations[n]
        ) = MD.Generate_LJ_NVT_MolecularDynamics_Trajectory(d,m,LBox,kT,run_time,
                                                         local_list_of_starting_configurations[n],
                                                         time_step=time_step,
                                                         number_of_time_steps_between_stored_configurations
                                                         = number_of_time_steps_between_stored_configurations,
                                                         number_of_time_steps_between_velocity_resets
                                                         = number_of_time_steps_between_velocity_resets,
                                                         start_from_stable_grid
                                                         = start_from_stable_grid
                                                        )

        # append trajectories to corresponding ensemble lists
        t_tr_ens.append(t_tr)
        x_tr_ens.append(x_tr)
        vx_tr_ens.append(vx_tr)
        y_tr_ens.append(y_tr)
        vy_tr_ens.append(vy_tr)
        uKin_tr_ens.append(uKin_tr)
        uPot_tr_ens.append(uPot_tr)
        pKin_tr_ens.append(pKin_tr)
        pPot_tr_ens.append(pPot_tr)
        pHyper_tr_ens.append(pHyper_tr)
    
    if debug:
        print("")
    t_tr_ens = np.array(t_tr_ens)
    x_tr_ens = np.array(x_tr_ens)
    y_tr_ens = np.array(y_tr_ens)
    vx_tr_ens = np.array(vx_tr_ens)
    vy_tr_ens = np.array(vy_tr_ens)
    uKin_tr_ens = np.array(uKin_tr_ens)
    uPot_tr_ens = np.array(uPot_tr_ens)
    pKin_tr_ens = np.array(pKin_tr_ens)
    pPot_tr_ens = np.array(pPot_tr_ens)
    pHyper_tr_ens = np.array(pHyper_tr_ens)
    
    return (t_tr_ens, x_tr_ens, y_tr_ens, vx_tr_ens, vy_tr_ens, 
            uPot_tr_ens, uKin_tr_ens, pPot_tr_ens, pKin_tr_ens, pHyper_tr_ens, 
            local_list_of_starting_configurations)








def get_results(d,m,system,run_time,method,ensemble=False,NTrajectories=10):
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
    
    
    if ensemble :
        (t_tr_ens, 
         x_tr_ens, 
         y_tr_ens, 
         vx_tr_ens, 
         vy_tr_ens, 
         uPot_tr_ens, 
         uKin_tr_ens, 
         pPot_tr_ens, 
         pKin_tr_ens, 
         pHyper_tr_ens, 
         local_list_of_starting_configurations) = Generate_Ensemble_of_LJ_NVT_MolecularDynamics_Trajectories(
             d,m,system,NTrajectories,run_time,
             number_of_time_steps_between_stored_configurations=1)
             
    
    else :
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
    
    
    if method == 'Compressibility':
        #Compute the compressibility
        compressibility = MD.Compressibility_from_pressure_fluctuations_in_NVT(d,m,
                                                                                    NParticles,
                                                                                    LBox,
                                                                                    kT,
                                                                                    pPot_tr, 
                                                                                    pHyper_tr, 
                                                                                    pKin_tr)
        
        return compressibility
    
    if method == 'MSD':
        
        if ensemble:
            msd_x_ens = []
            msd_y_ens = []
            
            for i in x_tr_ens:
                delta_t , msd_x = MD.MeanSquareDisplacements(t_tr_ens[0,:],i)
                msd_x_ens.append(msd_x)
                
            for i in y_tr_ens:
                delta_t , msd_y = MD.MeanSquareDisplacements(t_tr_ens[0,:],i)
                msd_y_ens.append(msd_y)
                
            msd_x_ens= np.array(msd_x_ens)
            msd_y_ens = np.array(msd_y_ens)
            
            msd_tot_ens = (msd_x_ens + msd_y_ens)/2
            msd_tot_ens = msd_tot_ens[:,:100]
            msd_mean = np.mean(msd_tot_ens)
            msd_std = np.std(msd_tot_ens)
            
            
            return msd_mean , msd_std
            
        else :
            #Compute the mean square displacement
            delta_t , msd_x = MD.MeanSquareDisplacements(t_tr, x_tr)
            delta_t , msd_y = MD.MeanSquareDisplacements(t_tr, y_tr)
        
            msd_tot = (msd_x+msd_y)/2
        
            #Take only when equilibrated
            msd_tot = msd_tot[:100]
            msd = np.mean(msd_tot)
        
        
            return msd
    
    if method == 'Pressure':
        P = pPot_tr + pKin_tr
        
        #Take only when equilibrated
        P = P[:100]
        P = np.mean(P)
        
        
        return P





def main(d,NParticles,sigma,epsilon,kT,rho,method,ensemble=False,NTrajectories=10):
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
    method : str
        Method used, propertiy to compute

    Returns
    -------
    result : Array
        CAN BE :
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


    if method == 'Compressibility':
        #Compute the compressibility
        compressibility = pool.starmap(
            get_results,
            [(d, m, (LBox_i, kT_j), run_time,method,ensemble,NTrajectories) for LBox_i, kT_j in couple]
            )
    
        #End of multiprocessing
        pool.close()
        
        #Convert to np.array for better data analysis
        compressibility = np.array(compressibility)
        #Reshape to have a square matrix
        compressibility = compressibility.reshape((len(rho),len(kT)))
        result = compressibility
        plot = result
    
    if method == 'MSD':
        #Compute the mean square displacement
        msd = pool.starmap(
            get_results,
            [(d, m, (LBox_i, kT_j), run_time,method,ensemble,NTrajectories) for LBox_i, kT_j in couple]
            )
        
        #End of multiprocessing
        pool.close()
        
        msd = np.array(msd)
        
        if ensemble:
            msd_mean = msd[:,0]
            msd_std = msd[:,1]
            msd_mean = msd_mean.reshape((len(rho),len(kT)))
            msd_std = msd_std.reshape((len(rho),len(kT)))
            
            result = (msd_mean,np.sqrt(msd_std))
            plot = msd_mean
        
        
        else:
            msd = msd.reshape((len(rho),len(kT)))
            result = msd
            plot = result
        
    if method == 'Pressure':
        #Compute the pressure
        P = pool.starmap(
            get_results,
            [(d, m, (LBox_i, kT_j), run_time,method,ensemble,NTrajectories) for LBox_i, kT_j in couple]
            )
        
        #End of multiprocessing
        pool.close()
        
        P = np.array(P)
        P = P.reshape((len(rho),len(kT)))
        result = P
        plot = result
        
    
    #Plot
    
    plt.imshow(plot,extent=(0.05,0.8,0.3,5),aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label(method)
    plt.xlabel('rho')
    plt.ylabel('kT')
    plt.title('Phase diagram')
    plt.savefig('phase_diagramm.png')
    
    return result
    
if __name__ == '__main__':
    
    #Dimension
    d=2
    #Number of particles
    NParticles=4
    
    #Number of trajectories
    NTrajectories = 2
    

    # Lennard-Jones
    sigma = 1
    epsilon = 1

    #Temperature and density
    kT = np.linspace(0.3,1,2)
    rho  = np.linspace(0.05,0.8,2)/sigma**d
    
    ##Get compressibility
    #compressibility = main(
    #    d,NParticles,sigma,epsilon,kT,rho,method='Compressibility')
    ##Save it to a .npy file
    #np.save('compressibility.npy',compressibility)
    
    # #Get MSD
    # msd_mean , msd_std = main(
    #     d,NParticles,sigma,epsilon,kT,rho,method='MSD',ensemble=True,NTrajectories=NTrajectories)
    # #Save it to a .npy file
    # np.save('msd_mean.npy',msd_mean)
    # np.save('msd_std.npy',msd_std)
    
    # #Get Pressure
    # P = main(
    #     d,NParticles,sigma,epsilon,kT,rho,method='Pressure')
    # #Save it to a .npy file
    # np.save('pressure.npy',P)










