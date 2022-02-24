import numpy as np
from scipy.special import gamma
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors, rc
import random



########## Plotting of conformations and animations of MD trajectories ###############################################

def PlotMDConf(x,y,xBox=(0,0),xpbc=False,yBox=(0,0),ypbc=False,m=1,c=0):
    """
    Plot 2d conformations of point particle systems
    
    x and y must be 1d arrays of equal length=NParticles containing the particles' x- and y-positions

    When xBox or yBox are specified the plot limits are set to these ranges. 
        Otherwise they are chosen such that all particles are shown.
        
    If xpbc or ypbc are True, positions are folded into xBox or yBox using periodic boundary conditions
    
    m specifies the particle mass, which is chosen to set the symbol size in proportion to sqrt(m). 
        m needs to be float point number (if all masses are equal) or a 1d array of length=NParticles
        
    c allows to set the particle color on a scale between 0 and 3.
        c needs to be float point number (if all colors are equal) or a 1d array of length=NParticles
    """
    
    # make sure, that we do not change the data outside the function
    x_show = np.copy(x)
    y_show = np.copy(y)
        
    (xmin,xmax,Lx) = BoxDimensions(xBox)
    if xpbc:
        # will lead to 'division by zero' without sensible xBox
        x_show = x - np.floor((x-xmin)/Lx)*Lx
    else:
        if xmax-xmin==0:   # without sensible xBox plot everything
            xmax = np.max(x)
            xmin = np.min(x)
        Lx = xmax-xmin
        x_show = np.copy(x)

    (ymin,ymax,Ly) = BoxDimensions(yBox)
    if ypbc:
        # will lead to 'division by zero' without sensible xBox
        y_show = y - np.floor((y-ymin)/Ly)*Ly
    else:
        if ymax-ymin==0:   # without sensible xBox plot everything
            ymax = np.max(y)
            ymin = np.min(y)
        Ly = ymax-ymin
        y_show = np.copy(y)

    ax = plt.axes()
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    ax.set_aspect(Lx/Ly)
    
    if isinstance(1.*m,float):
        size = 10*np.sqrt(m)*np.ones(len(x))
    else:
        size = 10*np.sqrt(m)
        
    colornorm = colors.Normalize(vmin = 0, vmax = 3)
    if isinstance(1.*c,float):
        c = c*np.ones(len(x))
        plot_colorbar=False
    else:
        plot_colorbar=True

    scat = ax.scatter(x_show,y_show,s=size,c=c, cmap=cm.magma, norm = colornorm)
    #
    # the following commands anticipate the animation of MD trajectories
    # where it is necessary to update the information passed to the scatter routine
    # 
    #scat.set_offsets(np.c_[xFold,yFold])       # ..., the offset function works with a list of [x,y] positions even though scatter asks for two separate arrays with the x- and y-positions, ... #
    #scat.set_sizes(m)           # changing the particles sizes (make little sense in clasical MD #)
    #scat.set_array(c)          # seems that this is enough to change the color! 

    if plot_colorbar:
        plt.colorbar(scat)

    return plt.show()


def PlotMDConfWithAssociated2dVectors(x,y,vx,vy,xBox=(0,0),xpbc=False,yBox=(0,0),ypbc=False,m=1,c=0):
    """
    Plot 2d conformations of point particle systems with associated vectors
    
    x and y must be 1d arrays of equal length=NParticles containing the particles' x- and y-positions

    vx and vy must be 1d arrays of equal length=NParticles containing the particles' x- and y-velocties or forces

    When xBox or yBox are specified the plot limits are set to these ranges. 
        Otherwise they are chosen such that all particles are shown.
        
    If xpbc or ypbc are True, positions are folded into xBox or yBox using periodic boundary conditions
    
    m specifies the particle mass, which is chosen to set the symbol size in proportion to sqrt(m). 
        m needs to be float point number (if all masses are equal) or a 1d array of length=NParticles
        
    c allows to set the particle color on a scale between 0 and 3.
        c needs to be float point number (if all colors are equal) or a 1d array of length=NParticles
    """
    
    # make sure, that we do not change the data outside the function
    x_show = np.copy(x)
    y_show = np.copy(y)
        
    (xmin,xmax,Lx) = BoxDimensions(xBox)
    if xpbc:
        # will lead to 'division by zero' without sensible xBox
        x_show = x - np.floor((x-xmin)/Lx)*Lx
    else:
        if xmax-xmin==0:   # without sensible xBox plot everything
            xmax = np.max(x)
            xmin = np.min(x)
        Lx = xmax-xmin
        x_show = np.copy(x)

    (ymin,ymax,Ly) = BoxDimensions(yBox)
    if ypbc:
        # will lead to 'division by zero' without sensible xBox
        y_show = y - np.floor((y-ymin)/Ly)*Ly
    else:
        if ymax-ymin==0:   # without sensible xBox plot everything
            ymax = np.max(y)
            ymin = np.min(y)
        Ly = ymax-ymin
        y_show = np.copy(y)

    ax = plt.axes()
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    ax.set_aspect(Lx/Ly)
    
    if isinstance(1.*m,float):
        size = 10*np.sqrt(m)*np.ones(len(x))
    else:
        size = 10*np.sqrt(m)
        
    colornorm = colors.Normalize(vmin = 0, vmax = 3)
    if isinstance(1.*c,float):
        c = c*np.ones(len(x))
        plot_colorbar=False
    else:
        plot_colorbar=True

    scat = ax.scatter(x_show,y_show,s=size,c=c, cmap=cm.magma, norm = colornorm)
    vec = ax.quiver(x_show,y_show, vx, vy,scale =20)
    #
    # the following commands anticipate the animation of MD trajectories
    # where it is necessary to update the information passed to the scatter routine
    # 
    #scat.set_offsets(np.c_[xFold,yFold])       # ..., the offset function works with a list of [x,y] positions even though scatter asks for two separate arrays with the x- and y-positions, ... #
    #scat.set_sizes(m)           # changing the particles sizes (make little sense in clasical MD #)
    #scat.set_array(c)          # seems that this is enough to change the color! 

    if plot_colorbar:
        plt.colorbar(scat)

    return plt.show()


def AddParticleTraces(ax,a_tr,b_tr,particles,lines=True,label=""):
    """
    Add ab-trajectories of selected particles to ax
    
    a_tr and b_tr may be 
        (NSteps) arrays of dimension 1 (typically for the time axis) or
        (NSteps x NParticles) arrays of dimension 2
    particles needs to be a tuple containing the numbers of the selected particles like (0,5,17)
        to select only one particle write (5,)
    """
    for i in particles:
        if a_tr.ndim==1: 
            a = a_tr        # time
        else:
            a = np.transpose(a_tr)[i]     # pick trace of particle i
        if b_tr.ndim==1: 
            b = b_tr        # time
        else:
            b = np.transpose(b_tr)[i]     # pick trace of particle i
        if lines:
            ax.plot(a,b,label=label)
        else:
            ax.scatter(a,b,label=label,s=1)
    return 


def AnimateMDRun(t_tr,x_tr,y_tr,xBox=(0,0),xpbc=False,yBox=(0,0),ypbc=False,m=1,c_tr=0,debug=False):
    """
    Animation of tajectories of 2d conformations of point particle systems
    
    x_tr and y_tr must be 1d arrays of equal length=NParticles containing the particles' x- and y-positions

    When xBox or yBox are specified the plot limits are set to these ranges. 
        Otherwise they are chosen such that all particles are shown.
        
    If xpbc or ypbc are True, positions are folded into xBox or yBox using periodic boundary conditions
    
    m specifies the particle mass, which is chosen to set the symbol size in proportion to sqrt(m). 
        m needs to be float point number (if all masses are equal) or a 1d array of length=NParticles
        
    c_tr allows to set the particle color on a scale between 0 and 3.
        c needs to be float point number (if all colors are equal) or a 2d array of the same dimensions as x_tr and y_tr
    """

    if isinstance(x_tr,np.ndarray):
        NParticles = x_tr.shape[1]
        NTimeSteps = x_tr.shape[0]
    else:
        NParticles = len(x_tr[0])
        NTimeSteps = len(x_tr)
    if debug:
        print("(NTimeSteps,NParticles) = ",x_tr.shape,NTimeSteps,NParticles)
        
    (xmin,xmax,Lx) = BoxDimensions(xBox)
    if xpbc:
        # will lead to 'division by zero' without sensible xBox
        x_tr_show = x_tr - np.floor((x_tr-xmin)/Lx)*Lx
    else:
        if xmax-xmin==0:   # without sensible xBox plot everything
            xmax = np.max(x_tr)
            xmin = np.min(x_tr)
        Lx = xmax-xmin
        x_tr_show = np.copy(x_tr)
    if debug:
        print("(xmin,xmax) = ",(xmin,xmax),Lx)
        print(x_tr_show[0][:min(5,NParticles)])

    (ymin,ymax,Ly) = BoxDimensions(yBox)
    if ypbc:
        # will lead to 'division by zero' without sensible xBox
        y_tr_show = y_tr - np.floor((y_tr-ymin)/Ly)*Ly
    else:
        if ymax-ymin==0:   # without sensible xBox plot everything
            ymax = np.max(y_tr)
            ymin = np.min(y_tr)
        Ly = ymax-ymin
        y_tr_show = np.copy(y_tr)
    if debug:
        print("(ymin,ymax) = ",(ymin,ymax),Ly)
        print(y_tr_show[0][:min(5,NParticles)])


    if isinstance(1.*m,float):
        size = 10*np.sqrt(m)*np.ones(NParticles)
    else:
        size = 10*np.sqrt(m)
    if debug:
        print("size = \n",size[:min(5,NParticles)])

        
    colornorm = colors.Normalize(vmin = 0, vmax = 3)
    if isinstance(1.*c_tr,float):
        c_tr = c_tr*np.ones((NTimeSteps,NParticles))
        plot_colorbar=False
    else:
        plot_colorbar=True
    if debug:
        print("c = \n",c_tr[0][:min(5,NParticles)])

    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    ax.set_aspect(Lx/Ly)
    colornorm = colors.Normalize(vmin = 0, vmax = 3)
    scat = ax.scatter([],[]) 
 
    def init():
        scat.set_array(c_tr[0]) # Needs to be set first for mappable
        if debug:
            print("set_array(c_tr[0])")
        scat.set_cmap(cm.magma)
        scat.set_norm(colornorm)
        scat.set_sizes(size)
        if debug:
            print("set_sizes(m)")
        if plot_colorbar:
            plt.colorbar(scat)
        plt.title("init")
        if debug:
            print("init")
        return scat

    def animate(i):
        t = t_tr[i]
        x = x_tr_show[i]
        y = y_tr_show[i]
        c = c_tr[i]
        scat.set_offsets(np.c_[x,y])
        scat.set_array(c)
        plt.title("t = "+str(round(t, 1)))
        return scat

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=NTimeSteps, interval=100)
    return anim



############### Setup ###########################################################################################


def BoxDimensions(xBox):
    """
    returns (xmin,xmax,LBox=xmax-xmin) for box dimensions defined either as
    xBox=(xmin,xmax) or xBox=LBox, in which case xmin is set to zero
    """
    if isinstance(xBox, (int, np.integer,float,np.float)):
        LBox = xBox
        xmin = 0
        xmax = xmin + LBox
    else:
        (xmin,xmax) = xBox
        LBox = xmax - xmin
    return (xmin,xmax,LBox)

def RandomPositions(xBox,NParticles):
    """
    Returns an array of NParticles random 1d positions 
    in the interval xBox = [xmin,xmax) or xBox * [0,1), if xBox is a float or integer
    """
    (xmin, xmax, LBox) = BoxDimensions(xBox)
    return xmin + LBox * np.random.random(NParticles)

def RandomVelocities(m,kT,ZeroCMMomentum=True):
    """
    Returns an NParticles array of random 1d velocites with <v^2> given by equipartition
    If ZeroCMMomentum is True, the random velocities are corrected for the center-of-mass velocity of the generated ensemble
    """
    vel = np.sqrt(12.*kT/m) * (np.random.random_sample(len(m))-0.5)
    vCM = 0
    if ZeroCMMomentum:
        M = 0
        for n in range(len(m)):
            M += m[n]
            vCM += m[n]*vel[n]
        vCM /= M
    return vel - vCM


def BattleLinePositions(xBox,NParticles):
    """
    Returns an array of NParticles 1d particle locations 
    in the interval xBox = [xmin,xmax) or xBox * [0,1), if xBox is a float or integer
    
    Half of the particles are located at xmin + LBox/4 and the other half at xmin + 3*LBox/4
    where LBox = xmax-xmin
    """
    (xmin, xmax, LBox) = BoxDimensions(xBox)
    return xmin + np.concatenate(( LBox/4*np.ones(int(NParticles/2)), 3*LBox/4*np.ones(int(NParticles/2)) ))

def BattleLineVelocities(m,kT):
    """
    Returns an NParticles x 2 array of random x-positions in the interval [0,1), with the y-positions confined to two narrow stripes
    """
    return np.sqrt(kT/m) * np.concatenate((np.ones(int(NParticles/2)), -np.ones(int(NParticles/2))))


def RandomHOPositions(xBox,k,kT):
    """
    Returns two NParticles array of random 1d rest and instanteneous positions for harmonic oscillators
    
    The rest positions are located in the interval xBox = [xmin,xmax) or xBox * [0,1), if xBox is a float or integer
    """
    (xmin, xmax, LBox) = BoxDimensions(xBox)
    x0 = xmin + LBox * np.random.random(len(k))
    x = x0 + np.sqrt(12.*kT/k) * (np.random.random_sample(len(k))-0.5)
    return (x0, x)


def GridPositionsIn2d(xBox,yBox,NParticles,debug=False):
    """
    Returns two arrays of stable x- and y-positions for NParticles in the interval 
        xBox = [xmin,xmax) or xBox * [0,1), if xBox is a float or integer
        yBox = [xmin,xmax) or xBox * [0,1), if xBox is a float or integer
    """
    n = np.ceil(np.sqrt(NParticles))            # number of particles in a row or column of our square grid
    (xmin, xmax, XBox) = BoxDimensions(xBox)
    ax = XBox / n
    (ymin, ymax, YBox) = BoxDimensions(yBox)
    ay = XBox / n
    
    if debug:
        print("Lattice constant: ",ax,ay)
    
    x = FoldIntoBox(xBox,xmin+ax*(np.arange(NParticles)+0.5))
    y = ymin+ ay*(np.arange(NParticles)//n+0.5)
        
    return x,y


def GridPositionsIn2dForDensityOne(sigma,NParticles,debug=False):
    """
    Returns two arrays of x- and y-positions for NParticles on a square shaped
    unstable square grid with lattice constant sigma
    """
    n = np.ceil(np.sqrt(NParticles))            # number of particles in a row or column of our square grid
    ax = sigma
    ay = sigma
    
    if debug:
        print("Lattice constant: ",ax,ay)
    
    x = ax*(np.arange(NParticles)%n+0.5)
    y = ay*(np.arange(NParticles)//n+0.5)
        
    return x,y


def StableGridPositionsIn2d(xBox,yBox,NParticles,debug=False):
    """
    Returns two arrays of stable x- and y-positions for NParticles in the interval 
        xBox = [xmin,xmax) or xBox * [0,1), if xBox is a float or integer
        yBox = [xmin,xmax) or xBox * [0,1), if xBox is a float or integer
    """
    n = np.ceil(np.sqrt(NParticles))            # number of particles in a row or column of our square grid
    (xmin, xmax, XBox) = BoxDimensions(xBox)
    ax = XBox / n
    (ymin, ymax, YBox) = BoxDimensions(yBox)
    ay = XBox / n
    
    if debug:
        print("Lattice constant: ",ax,ay)
    
    x = FoldIntoBox(xBox,xmin+ax*(np.arange(NParticles)+0.5*(np.arange(NParticles)//n%2) + 0.25))
    y = ymin+ ay*(np.arange(NParticles)//n+0.5)
        
    return x,y

def StableGridPositionsIn2dForDensityOne(sigma,NParticles,debug=False):
    """
    Returns two arrays of x- and y-positions for NParticles on a square shaped
    triangular grid with lattice constant sigma
    """
    n = np.ceil(np.sqrt(NParticles))            # number of particles in a row or column of our square grid
    ax = sigma
    ay = sigma
    
    if debug:
        print("Lattice constant: ",ax,ay)
    
    x = ax*(np.arange(NParticles)%n+0.5*(np.arange(NParticles)//n%2) + 0.25)
    y = ay*(np.arange(NParticles)//n+0.5)
        
    return x,y


def FoldIntoBox(xBox,x):
    (xmin, xmax, LBox) = BoxDimensions(xBox)
    return x - np.floor((x-xmin)/LBox)*LBox

def MinimumImage(xBox,dx):
    (xmin, xmax, LBox) = BoxDimensions(xBox)
    return dx - np.rint(dx/LBox)*LBox

def BinCenters(BinEdges):
    return (0.5*BinEdges+np.roll(0.5*BinEdges,1))[1:]

############### Properties ###########################################################################################

def ParticleKineticEnergies(m, vel):
    """
    instantaneous kinetic energies
        m is the particle mass and can be 
            a scalar for a single particle or 
            an array of length NParticles
            
        v represents one Cartesian component of the instanteneous velocities and can be
            a scalar for a single particle or
            an array of length NParticles for a single conformation
            an array of shape (NTimeSteps,NParticles) for a trajectory
            an array of shape (NTrajectories,NTimeSteps,NParticles) for an ensemble of trajectories
            
        UKin has the same shape as v     
    """
    v = np.array(vel)
    return 0.5*m*v**2

def TotalKineticEnergy(m,v):
    """
    Returns the total instantaneous kinetic energy UKin for one conformation of a system
        m is the particle mass and can be 
            a scalar for a single particle or 
            an array of length NParticles
            
        v represents one Cartesian component of the instanteneous velocities and can be
            a scalar for a single particle or
            an array of length NParticles for a single conformation
            an array of shape (NTimeSteps,NParticles) for a trajectory
            an array of shape (NTrajectories,NTimeSteps,NParticles) for an ensemble of trajectories
            
        UKin is
            a scalar for a single particle or conformation
            an array of shape (NTimeSteps) for a trajectory
            an an array of shape (NTrajectories,NTimeSteps) for an ensemble of trajectories
     """
    return np.sum(ParticleKineticEnergies(m, v),axis=-1)


def KineticTemperature(m,v):
    """
    Returns the the instantaneous kinetic temperature corresponding to TotalKineticEnergy(m,v)
    """
    return 2*TotalKineticEnergy(m,v)/np.shape(v)[-1]


def ParticleLinearMomentum(m,v):
    return m*v

def TotalLinearMomentum(m,v):
    return np.sum(ParticleLinearMomentum(m,v),axis=-1)

def TimeAverage(a_tr):
    return np.sum(a_tr,axis=0)/len(a_tr)


####################### Harmonic oscillator

def HO_PotentialEnergies(k,r0,r):
    """
    Returns instantaneous HO potential energies UPot
        k is the HO spring constant and can be
            be a scalar for a single particle or 
            an array of length NParticles
            
        r0 represents one Cartesion component of the rest position and can be 
            a scalar for a single particle or 
            an array of length NParticles

        r represents one Cartesion component of the instanteneous positions and can be
            a scalar for a single particle or
            an array of length NParticles for a single conformation
            an array of shape (NTimeSteps,NParticles) for a trajectory
            an array of shape (NTrajectories,NTimeSteps,NParticles) for an ensemble of trajectories
            
        UPot has the same shape as r     
    """
    return 0.5*k*(r-r0)**2

def Total_HO_PotentialEnergy(k,r0,r):
    """
    Returns the total instantaneous HO potential energy UPot for one conformation of a system composed of HO
        k is the HO spring constant and can be
            be a scalar for a single particle or 
            an array of length NParticles
            
        r0 represents one Cartesion component of the rest position and can be 
            a scalar for a single particle or 
            an array of length NParticles

        r represents one Cartesion component of the instanteneous positions and can be
            a scalar for a single particle or
            an array of length NParticles for a single conformation
            an array of shape (NTimeSteps,NParticles) for a trajectory
            an array of shape (NTrajectories,NTimeSteps,NParticles) for an ensemble of trajectories
            
        UPot is
            a scalar for a single particle or conformation
            an array of shape (NTimeSteps) for a trajectory
            an an array of shape (NTrajectories,NTimeSteps) for an ensemble of trajectories
     """
    return np.sum(HO_PotentialEnergies(k,r0,r),axis=-1)

def HO_ConfigurationalTemperature(k,r0,r):
    """
    Returns the the instantaneous configurational temperature corresponding to Total_HO_PotentialEnergy(k,r0,r)
    """
    Teff = 2*Total_HO_PotentialEnergy(k,r0,r)/np.shape(r)[-1]
    return Teff


def U_HO(d,spring_constant,rest_pos,pos):
    """
    Potential energy for a harmonic oscillator in d dimensions
    
    d is the embedding dimension. Needed to distinguish the case of 2 HO in 1d from 1 HO in 2d.
    
    k is the spring constant and can be
        a scalar for a single HO or if all HO have the same spring constant
        an NParticles-array if the HO have different k
        
    rest_pos are the rest positions 
        in d = 1 rest_pos = x0
        in d>1 rest_pos have to be of the form rest_pos = (x0,y0) or pos = [x0,y0]
        where x0 and y0 can be 
            a scalar for a single HO or if all HO have the same rest position
            an NParticles-array for a system composed of H0 with different rest position

    pos are the instanteneous positions 
        in d = 1 pos = x
        in d>1 pos have to be of the form pos = (x,y) or pos = [x,y]
        where x and y can be    
            a scalar for a single HO
            an NParticles-array for a system composed of several HO
    
    The function returns u = 0.5*spring_constant*(pos - rest_pos)**2 where f has the same format as pos and rest_pos
    """
    k = np.array(spring_constant) #.transpose() wouldn't change anything
    r = np.array(pos).transpose() # NParticles array of scalar(d=1) or vector (d>1) positions
    r0 = np.array(rest_pos).transpose() # same for rest positions
    delta_r_sqr = (r-r0)**2  # NParticles array of Cartesian components of squared spring extension
    if d > 1:
        delta_r_sqr = np.sum(delta_r_sqr,axis=-1)         # add up Cartesian components in d>1
    u = 0.5*k*delta_r_sqr # NParticles array of spring energies
    return u.transpose()

def f_HO(d,spring_constant,rest_pos,pos):
    """
    Restoring force for a harmonic oscillator in d dimensions
    
    d is the embedding dimension. 
        Kept for consistency with the general case, where one needs to distinguish the case of 2 HO in 1d from 1 HO in 2d.
    
    k is the spring constant and can be
        a scalar for a single HO or if all HO have the same spring constant
        an NParticles-array if the HO have different k
        
    rest_pos are the rest positions 
        in d = 1 rest_pos = x0
        in d>1 rest_pos have to be of the form rest_pos = (x0,y0) or pos = [x0,y0]
        where x0 and y0 can be 
            a scalar for a single HO or if all HO have the same rest position
            an NParticles-array for a system composed of H0 with different rest position

    pos are the instanteneous positions 
        in d = 1 pos = x
        in d>1 pos have to be of the form pos = (x,y) or pos = [x,y]
        where x and y can be    
            a scalar for a single HO
            an NParticles-array for a system composed of several HO
    
    The function returns f = - spring_constant*(pos - rest_pos) where f has the same format as pos and rest_pos
    """
    k = np.array(spring_constant) #.transpose() wouldn't change anything
    r = np.array(pos).transpose() # NParticles array of scalar(d=1) or vector (d>1) positions
    r0 = np.array(rest_pos).transpose()
#    delta_r_sqr = (r-r0)**2  # NParticles array of Cartesian components of squared spring extension
#    if d > 1:
#        delta_r_sqr = np.sum(delta_r_sqr,axis=-1)         # add up Cartesian components in d>1
    f = -k*(r-r0) # calculate forces particle for particle
    return f.transpose() # in d>1 transform back to fx- and fy-arrays


#### maybe FENE functions should be rewritten following the LJ example, which properly handles all levels of configurations, trajectories, ensembles, ....
def U_FENE(d,spring_constant,max_elongation,rest_pos,pos):
    """
    Potential energy for a FENE spring in d dimensions
    
    d is the embedding dimension. Needed to distinguish the case of 2 FENE springs in 1d from 1 FENE spring in 2d.
    
    k is the spring constant and can be
        a scalar for a single HO or if all HO have the same spring constant
        an NParticles-array if the HO have different k
        
    max_elongations is the maximal spring elongation and can be
        a scalar for a single HO or if all HO have the same maximal elongations
        an NParticles-array if the HO have different maximal elongations

    rest_pos are the rest positions 
        in d = 1 rest_pos = x0
        in d>1 rest_pos have to be of the form rest_pos = (x0,y0) or pos = [x0,y0]
        where x0 and y0 can be 
            a scalar for a single HO or if all HO have the same rest position
            an NParticles-array for a system composed of H0 with different rest position

    pos are the instanteneous positions 
        in d = 1 pos = x
        in d>1 pos have to be of the form pos = (x,y) or pos = [x,y]
        where x and y can be    
            a scalar for a single HO
            an NParticles-array for a system composed of several HO

    The function returns f = - spring_constant*(pos - rest_pos)/(1-(r/L)**2) where f has the same format as pos and rest_pos
    """
    k = np.array(spring_constant)       
    Lmax2 = np.array(max_elongation)**2     
    r = np.array(pos).transpose()       # NParticles array of scalar(d=1) or vector (d>1) positions
    r0 = np.array(rest_pos).transpose() # single scalar(d=1) or vector (d>1) rest position or corresponding NParticles array
    delta_r_sqr = (r-r0)**2  # NParticles array of Cartesian components of squared spring extension
    if d > 1:
        delta_r_sqr = np.sum(delta_r_sqr,axis=-1)         # add up Cartesian components in d>1
    relative_L2 = delta_r_sqr/Lmax2
#    relative_L2 = np.min([L2/Lmax2,0.99*np.ones(L2.size)],axis=0)
    u = -0.5*Lmax2*np.log(1-relative_L2)
    return u.transpose()

def f_FENE(d,spring_constant,max_elongation,rest_pos,pos):
    """
    Restoring force for a FENE spring in d dimensions
    
    d is the embedding dimension. Needed to distinguish the case of 2 HO in 1d from 1 HO in 2d.
    
    k is the spring constant and can be
        a scalar for a single HO or if all HO have the same spring constant
        an NParticles-array if the HO have different k
        
    max_elongations is the maximal spring elongation and can be
        a scalar for a single HO or if all HO have the same maximal elongations
        an NParticles-array if the HO have different maximal elongations

    rest_pos are the rest positions 
        in d = 1 rest_pos = x0
        in d>1 rest_pos have to be of the form rest_pos = (x0,y0) or pos = [x0,y0]
        where x0 and y0 can be 
            a scalar for a single HO or if all HO have the same rest position
            an NParticles-array for a system composed of H0 with different rest position

    pos are the instanteneous positions 
        in d = 1 pos = x
        in d>1 pos have to be of the form pos = (x,y) or pos = [x,y]
        where x and y can be    
            a scalar for a single HO
            an NParticles-array for a system composed of several HO

    The function returns f = - spring_constant*(pos - rest_pos)/(1-(r/L)**2) where f has the same format as pos and rest_pos
    """
    k = np.array(spring_constant)       
    Lmax2 = np.array(max_elongation)**2     
    r = np.array(pos).transpose()       # NParticles array of scalar(d=1) or vector (d>1) positions
    r0 = np.array(rest_pos).transpose() # single scalar(d=1) or vector (d>1) rest position or corresponding NParticles array
    delta_r_sqr = (r-r0)**2  # NParticles array of Cartesian components of squared spring extension
    if d > 1:
        delta_r_sqr = np.sum(delta_r_sqr,axis=-1)         # add up Cartesian components in d>1
    relative_L2 = delta_r_sqr/Lmax2
#    relative_L2 = np.min([L2/Lmax2,0.99*np.ones(L2.size)],axis=0)
    return -k/(1-relative_L2)*(r-r0).transpose() # transpose to go back to (fx,fy) format


############## Lennard-Jones

def U_LJ(d,epsilon,sigma,distance_vector):
    """
    Lennard-Jones potential energy in d dimensions
    
    d is the embedding dimension. Needed to distinguish the case of 2 1d distance vectors from 1 2d distance vector.
    
    epsilon is the energy scale of the LJ potential and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors
        
    sigma is the interaction range and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors

    distance_vector are the instanteneous distances 
        in d = 1 distance_vector = delta_x
        in d>1 distance_vector has to be of the form distance_vector = (delta_x,delta_x) or distance_vector = [delta_x,delta_x]
        where delta_x and delta_y can be    
            a scalar for a single interaction
            an array for several interactions to be evaluated simultaneously

    The function returns 4*epsilon*((sigma/r)**(-12)-(sigma/r)**-6) where u has the same format as delta_x
    """
    eps = np.array(epsilon)       
    sig = np.array(sigma)       
    delta_r = np.array(distance_vector)                   # array of scalar(d=1) or vector (d>1) distances
    delta_r_sqr = delta_r**2                              # array of Cartesian components of squared distances
    if d > 1:
        delta_r_sqr = np.sum(delta_r_sqr,axis=0)         # add up Cartesian components in d>1
    relative_inverse_squared_distance = sigma**2/delta_r_sqr
    u = 4*epsilon*(relative_inverse_squared_distance**6-relative_inverse_squared_distance**3)
    return u

def f_LJ(d,epsilon,sigma,distance_vector,debug=False):
    """
    Lennard-Jones force in d dimensions
    
    d is the embedding dimension. Needed to distinguish the case of 2 1d distance vectors from 1 2d distance vector.
    
    epsilon is the energy scale of the LJ potential and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors
        
    sigma is the interaction range and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors

    distance_vector are the instanteneous distances 
        in d = 1 distance_vector = delta_x
        in d>1 distance_vector has to be of the form distance_vector = (delta_x,delta_x) or distance_vector = [delta_x,delta_x]
        where delta_x and delta_y can be    
            a scalar for a single interaction
            an array for several interactions to be evaluated simultaneously

    The function returns -24*epsilon*(2*(sigma/r)**(-12)-(sigma/r)**-6) * distance_vector/r**2 
        where f has the same format as distance_vector
        and r = |distance_vector|
    """
    eps = np.array(epsilon)       
    sig = np.array(sigma)       
    delta_r = np.array(distance_vector)       # array of scalar(d=1) or vector (d>1) distances
    if debug: 
        print(delta_r)
    delta_r_sqr = delta_r**2                              # array of Cartesian components of squared distances
    if d > 1:
        delta_r_sqr = np.sum(delta_r_sqr,axis=0)         # add up Cartesian components in d>1
    relative_inverse_squared_distance = sigma**2/delta_r_sqr
    f = 24*epsilon*(2*relative_inverse_squared_distance**6-relative_inverse_squared_distance**3)/delta_r_sqr*delta_r
    return f


def LJ_forces_as_a_function_of_positions(d,epsilon,sigma,LBox,r,debug=False):
    """
    returns the LJ force acting on each particle as a function of the positions of all particles
    """
    r = np.array(r)
    N = r.shape[-1]
    f = 0.*np.copy(r) #initialise force array with the same shape as position array
    
    if debug:
        print(N)
    
    for k in range(1,N):
        delta_r_pair = MinimumImage(LBox,r-np.roll(r,k,axis=-1))
        fpair = f_LJ(d,epsilon,sigma,delta_r_pair)
        f += fpair
        f -= np.roll(fpair,-k,axis=-1)
    return f/2
    # normalize by 2, because each force is counted twice


def LJ_energies_as_a_function_of_positions(d,epsilon,sigma,LBox,r,debug=False):
    """
    returns the LJ interaction energy for each particle as a function of the positions of all particles
    """
    r = np.array(r)
    if d==1:
        u = 0.*np.copy(r)     # initialise energy array with the same shape as the d=1 position array
    else:
        u = 0.*np.copy(r[0])  # initialise energy array with the same shape as the first Cartesian component
                             # of the position array
    N = r.shape[-1]
    
    if(debug):
        print(N,NParticles,epsilon,sigma)
    
    for k in range(1,N):
        delta_r_pair = MinimumImage(LBox,r-np.roll(r,k,axis=-1))
        upair = U_LJ(d,epsilon,sigma,delta_r_pair)
        u += upair/2
        u += np.roll(upair,-k,axis=-1)/2
    return u/2
    # normalize by 2, because each interaction is counted twice
    
def LJ_energy_as_a_function_of_positions(d,epsilon,sigma,LBox,r):
    """
    returns the total LJ interaction energy as a function of the positions of all particles
    """
    return np.sum(LJ_energies_as_a_function_of_positions(d,epsilon,sigma,LBox,r),axis=-1)



def Virial_w_LJ(d,epsilon,sigma,distance_vector,debug=False):
    """
    Lennard-Jones virial in d dimensions
    
    d is the embedding dimension. Needed to distinguish the case of 2 1d distance vectors from 1 2d distance vector.
    
    epsilon is the energy scale of the LJ potential and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors
        
    sigma is the interaction range and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors

    distance_vector are the instanteneous distances 
        in d = 1 distance_vector = delta_x
        in d>1 distance_vector has to be of the form distance_vector = (delta_x,delta_x) or distance_vector = [delta_x,delta_x]
        where delta_x and delta_y can be    
            a scalar for a single interaction
            an array for several interactions to be evaluated simultaneously

    The function returns -24*epsilon*(2*(sigma/r)**(-12)-(sigma/r)**-6) * distance_vector/r**2 
        where f has the same format as distance_vector
        and r = |distance_vector|
    """
    eps = np.array(epsilon)       
    sig = np.array(sigma)       
    delta_r = np.array(distance_vector)       # array of scalar(d=1) or vector (d>1) distances
    if debug: 
        print(delta_r)
    delta_r_sqr = delta_r**2                              # array of Cartesian components of squared distances
    if d > 1:
        delta_r_sqr = np.sum(delta_r_sqr,axis=0)         # add up Cartesian components in d>1
    relative_inverse_squared_distance = sigma**2/delta_r_sqr
    w = 4*epsilon*(-12*relative_inverse_squared_distance**6 + 6*relative_inverse_squared_distance**3)
    return w


def LJ_virial_pressure_as_a_function_of_positions(d,epsilon,sigma,LBox,r,debug=False):
    """
    returns the contribution to the scalar virial pressure resulting from LJ interactions as a function of the particle positions
    """
    r_array = np.array(r)
    N = r_array.shape[-1]
    
    # initialise pressure:
    # one scalar value per system, i.e. same shape as adding up all x-coordinates per system
    if d==1:
        P = 0*np.copy(np.sum(r_array,axis=-1))     
    else:
        P = 0*np.copy(np.sum(r_array[0],axis=-1))  
    
    if debug:
        print("N:",N)
        print("P.shape:",P.shape)
        print("P:",P)
    
    for k in range(1,N):
        delta_r_pair = MinimumImage(LBox,r_array-np.roll(r_array,k,axis=-1))
        wpair = Virial_w_LJ(d,epsilon,sigma,delta_r_pair)
        
        P -= np.sum(wpair,axis=-1)
        
    return P/2/LBox**d/d    
    # normalize by 2, because each interaction is counted twice


def HyperVirial_x_LJ(d,epsilon,sigma,distance_vector,debug=False):
    """
    Lennard-Jones hyper virial in d dimensions
    
    d is the embedding dimension. Needed to distinguish the case of 2 1d distance vectors from 1 2d distance vector.
    
    epsilon is the energy scale of the LJ potential and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors
        
    sigma is the interaction range and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors

    distance_vector are the instanteneous distances 
        in d = 1 distance_vector = delta_x
        in d>1 distance_vector has to be of the form distance_vector = (delta_x,delta_x) or distance_vector = [delta_x,delta_x]
        where delta_x and delta_y can be    
            a scalar for a single interaction
            an array for several interactions to be evaluated simultaneously

    The function returns -24*epsilon*(2*(sigma/r)**(-12)-(sigma/r)**-6) * distance_vector/r**2 
        where f has the same format as distance_vector
        and r = |distance_vector|
    """
    eps = np.array(epsilon)       
    sig = np.array(sigma)       
    delta_r = np.array(distance_vector)       # array of scalar(d=1) or vector (d>1) distances
    if debug: 
        print(delta_r)
    delta_r_sqr = delta_r**2                              # array of Cartesian components of squared distances
    if d > 1:
        delta_r_sqr = np.sum(delta_r_sqr,axis=0)         # add up Cartesian components in d>1
    relative_inverse_squared_distance = sigma**2/delta_r_sqr
    x = 4*epsilon*(144*relative_inverse_squared_distance**6 - 36*relative_inverse_squared_distance**3)
    return x

def LJ_hyper_virial_as_a_function_of_positions(d,epsilon,sigma,LBox,r,debug=False):
    """
    returns the contribution to the scalar hyper virial resulting from LJ interactions as a function of the particle positions
    """
    r_array = np.array(r)
    N = r_array.shape[-1]
    
    # initialise pressure:
    # one scalar value per system, i.e. same shape as adding up all x-coordinates per system
    if d==1:
        P = 0*np.copy(np.sum(r_array,axis=-1))     
    else:
        P = 0*np.copy(np.sum(r_array[0],axis=-1))  
    
    if debug:
        print("N:",N)
        print("P.shape:",P.shape)
        print("P:",P)
    
    for k in range(1,N):
        delta_r_pair = MinimumImage(LBox,r_array-np.roll(r_array,k,axis=-1))
        xpair = HyperVirial_x_LJ(d,epsilon,sigma,delta_r_pair)
        
        P += np.sum(xpair,axis=-1)
        
    return P/2/LBox**d/d**2



def KineticPressure_as_a_function_of_velocities(d,LBox,m,v,debug=False):
    """
    returns the kinetic contribution to the virial pressure as a function of the particle velocities
    """
    v = np.array(v)
    N = v.shape[-1]
    V = LBox**d
    
    if debug:
        print(N,d,V)
    
    if d==1:
        # sum over particles
        P = np.sum(m*v*v/d/V,axis=-1)   
    else:
        # sum over particles and dimensions
        if debug:
            print(np.sum(m*v*v/d/V,axis=-1).shape)
        P = np.sum( np.sum(m*v*v/d/V,axis=-1), axis=0)       
    return P


def Compressibility_from_pressure_fluctuations_in_NVT(d,m,NParticles,LBox,kT,pPot, pHyper, pKin=0):
    """
    Returns an estimate of the compressibility based on the analysis of the fluctuations of 
    the virial pressure in the canonical ensemble
    """
    if isinstance(pKin,np.ndarray):
        beta_T = 1./( 2*NParticles*kT/d/LBox**d - np.var(pPot+pKin,axis=-1)*LBox**d/kT + np.mean(pPot+pKin+pHyper,axis=-1) )
    else:
        beta_T = 1./(                           - np.var(pPot     ,axis=-1)*LBox**d/kT + np.mean(pPot     +pHyper,axis=-1) )
        
    if pPot.ndim==1:
        # data for one trajectories
        return beta_T
    elif pPot.ndim==2:
        # data for an ensemble of trajectories
        NTrajectories = beta_T.size
        return np.mean(beta_T), np.std(beta_T)/np.sqrt(NTrajectories)
    
    
#### Structure and dynamics
    
def UnitHyperSphereSurface(d): return 2.*np.pi**(d/2.)/gamma(d/2.)
def UnitHyperSphereVolume(d): return np.pi**(d/2.)/gamma(d/2.+1)
    
def Radial_distribution_function(d,LBox,pos,r_range=(0.0,5.0),bins=50,debug=False):
    """
    returns the pair correlation function as a function of the positions of all particles
    """

    
    r = np.array(pos)
    
    if r.ndim>d:
        # data for a trajectory or an ensemble of trajectories
        NParticles = r.shape[-1]
        NDataSets = r.size//d//NParticles
        
    elif r.ndim==d:
        # data for an individual snapshot        
        NParticles = r.shape[-1]
        NDataSets = 1

    else:
        print("provided coordinates are dimension ",r.ndim)
        print("expected dimension ",d)
        return
        
    rho = (NParticles-1)/LBox**2
    # this is the density of all the OTHER particles, which a test particle can see
        
    if debug:
        print(NParticles, "particles in one configuration")
        print(NDataSets, "data sets")
        print("r_range",r_range)
        print("bins",bins)
    
    hist = np.zeros(bins)
    
    for k in range(1,NParticles):
        delta_r_pair_vectors = MinimumImage(LBox,r-np.roll(r,k,axis=-1))
        delta_r_pair_sqr = delta_r_pair_vectors**2              # array of Cartesian components of squared distances
        if d > 1:
            delta_r_pair_sqr = np.sum(delta_r_pair_sqr,axis=0)         # add up Cartesian components in d>1
        delta_r_pair = np.sqrt(delta_r_pair_sqr)
        hist_k, bin_edges = np.histogram(delta_r_pair, bins, r_range, normed=None, weights=None, density=None)
        hist += hist_k
        
    if debug:
        print(np.sum(hist)," pair distances")
        print(NParticles*(NParticles-1)*NDataSets," expected")

    # expected number of pair distances in spherical shell
    normalization = (UnitHyperSphereSurface(d)*BinCenters(bin_edges)**(d-1)*(r_range[1]-r_range[0])/bins*
                     # bin volume
                     rho*
                     # density
                     NDataSets*NParticles
                     # number of test particles = number of datasets * number of particles
                    )
        
    if debug:
        print(np.sum(normalization),"expected pair distances in an ideal gas")

    return hist/normalization, bin_edges  
    
    
def Number_of_nearest_neighbors(d,LBox,pos,r_max,debug=False):
    """
    returns for each particle and each snapshot the number of other particles found within a radius of r_max
    """
    r = np.array(pos)
    
    if r.ndim>d:
        # data for a trajectory or an ensemble of trajectories
        NParticles = r.shape[-1]
        NDataSets = r.size//d//NParticles
        
    elif r.ndim==d:
        # data for an individual snapshot        
        NParticles = r.shape[-1]
        NDataSets = 1

    else:
        print("provided coordinates are dimension ",r.ndim)
        print("expected dimension ",d)
        return
    
    if d==1:
        # output has the same format as 1d position information: 
        # instead of a float indicating a particle position we now have an integer indicating the number of close-by neighbors
        NNN = 0*np.copy(r).astype(int)
    else:
        # output has the same format as 1d position information: 
        # instead of a float indicating a particle position we now have an integer indicating the number of close-by neighbors
        NNN = 0*np.copy(r[0]).astype(int)
                
    if debug:
        print(NParticles, "particles in one configuration")
        print(NDataSets, "data sets")
        print("r_max",r_max)
    
    for k in range(1,NParticles):
        delta_r_pair_vectors = MinimumImage(LBox,r-np.roll(r,k,axis=-1))
        delta_r_pair_sqr = delta_r_pair_vectors**2              # array of Cartesian components of squared distances
        if d > 1:
            delta_r_pair_sqr = np.sum(delta_r_pair_sqr,axis=0)         # add up Cartesian components in d>1

        pair_in_range = np.round((np.sign(r_max**2-delta_r_pair_sqr)+1.0)/2.0).astype(int)
        
        NNN += pair_in_range

    return NNN


def CellOccupancy(NCells,xx,yy,xBox,yBox,x_pbc=False,y_pbc=False,debug=False):
    '''
        Returns the occupancy of NCells x NCells cells

        xBox = (xmin,xmax) denotes the simulation box
        xBox = LBox corresponds to a box of (0,LBox)

        if x_pbc==True
            particles are sorted according to their folded positions,
            where the box position is randomly shifted to break Gallilein invariance
        else:
            particles inside the box are sorted according to their absolute positions
    
        the same rules apply in y-direction
        
        The function can be applied to individual conformations, trajectories and ensembles
            and returns correspondingly shaped arrays
    '''
    
    def individualCellOccupancy(NCells,xx,yy,xBox,yBox,x_pbc=False,y_pbc=False,debug=False):
        # sort particles into cells

        #Keep cells approximately square
        (xmin, xmax, XBox) = BoxDimensions(xBox)
        (ymin, ymax, YBox) = BoxDimensions(yBox)
    
        LBox = np.sqrt(XBox*YBox)
        NCellsX = np.int(np.round(NCells*XBox/LBox))
        NCellsY = np.int(np.round(NCells*YBox/LBox))
    
        NParticles = len(xx)
        # sort particles into cells
        x = xx.copy()
        if x_pbc:
            x += XBox/NCellsX*(np.random.random()-0.5) # to break Gallilean invariance #
            x -= np.floor(x/XBox)*XBox # fold into box / minimum image convention for collisions #
        y = yy.copy() 
        if y_pbc:
            y += YBox/NCellsY*(np.random.random()-0.5) # to break Gallilean invariance #
            y -= np.floor(y/YBox)*YBox
        # cell numbers corresponding to x- and y-coordinates
        i = np.floor(NCellsX*x/XBox).astype(int)
        j = np.floor(NCellsY*y/YBox).astype(int)
        # combine into absolute cell number
        row = i + NCellsX*j
        if debug:
            print("row = ",row)
        # particle numbers
        col = np.array(range(NParticles))
        # combine into projector from particles to cells
        data = np.ones(NParticles)
        sparse_cell_projector = sparse.coo_matrix((data,(row,col)),shape=(NCellsX*NCellsY,NParticles))
        if debug:
            print("projector = ",sparse_cell_projector.todense())
        
        return sparse_cell_projector.A.sum(axis=1) 

    
    # vectorize application of individualCellOccupancy
    if xx.ndim==1:
        nOcc = individualCellOccupancy(NCells,xx,yy,xBox,yBox,x_pbc,y_pbc,debug)
    elif xx.ndim==2:
        # trajectory
        nOcc = []
        for i in range(xx.shape[0]):
            nOcc.append(individualCellOccupancy(NCells,xx[i],yy[i],xBox,yBox,x_pbc,y_pbc,debug))
    elif xx.ndim==3:
        # trajectory
        nOcc = []
        for i in range(xx.shape[0]):
            nOcc_tr = []
            for j in range(xx.shape[1]):
                nOcc_tr.append(individualCellOccupancy(NCells,xx[i,j],yy[i,j],xBox,yBox,x_pbc,y_pbc,debug))
            nOcc.append(nOcc_tr)
    else:
        print("CellOccupancy_vec not defined beyond the ensemble level")
    return np.array(nOcc)


def MeanSquareDisplacements(t_tr,x_tr):
    """
    Returns the particle and time average mean-square displacement < (x(t)-x(0))**2 > 
    in one Cartesian direction for a trajectory of x- or y-positions
    """
    
    NParticles = x_tr.shape[-1]
    length_of_x_tr = x_tr.shape[-2]
    length_of_t_tr = t_tr.shape[-1]
    local_x_tr = np.copy(x_tr)

    if x_tr.ndim>2:
        # data for an ensemble of trajectories
        NTrajectories = x_tr.shape[-3]
        local_x_tr = local_x_tr.transpose(1,0,2)
        # so that the time axis is always the first axis (or rather axis=0)
    else:
        NTrajectories = 1
        
    msd = []
    delta_t = []
    
    for n in range(1,length_of_x_tr//2):
        
        n_t = n*length_of_t_tr//length_of_x_tr  # because sometimes configurations are not stored for each time step
        if t_tr.ndim==1:
            # data for one trajectories
            delta_t.append(t_tr[n_t]-t_tr[0])
        elif t_tr.ndim==2:
            # data for an ensemble of trajectories
            delta_t.append(t_tr[0,n_t]-t_tr[0,0])
        delta_x2 = ( local_x_tr - np.roll(local_x_tr,-n,axis=0) )**2
        msd.append(np.mean(delta_x2[:length_of_x_tr-n]))
        
    return np.array(delta_t), np.array(msd)    
    
    
    
############### Dynamics I: Streaming ##########################################################################

def BallisticStreamingStep(pos,vel,delta_t,g=0):
    return pos + delta_t * np.array(vel),vel


def HarmonicOscillatorTimeStep(m,k,rest_pos,pos,vel,delta_t):
    omega = np.sqrt(k/m)
    new_pos = np.array(rest_pos) + np.cos(omega*delta_t)*(np.array(pos)-np.array(rest_pos)) + np.sin(omega*delta_t)/omega*np.array(vel)
    new_vel = -np.sin(omega*delta_t)*omega*(np.array(pos)-np.array(rest_pos)) + np.cos(omega*delta_t)*np.array(vel)
    return new_pos, new_vel

############### Dynamics II: Collisions ########################################################################

def SortParticlesIntoGrid(NCells,LBox,pos,x_pbc=True,y_pbc=True,debug=False):
    cells = [[set() for _ in range(NCells)]for _ in range(NCells)] # list of empty sets #
    x = 1.*pos[0] 
    nSorted = 0
    if x_pbc:
        x += LBox/NCells*(np.random.random()-0.5) # to break Gallilean invariance #
        x -= np.floor(x/LBox)*LBox # fold into box / minimum image convention for collisions #
    y = 1.*pos[1] 
    if y_pbc:
        y += LBox/NCells*(np.random.random()-0.5) # to break Gallilean invariance #
        y -= np.floor(y/LBox)*LBox
    for n in range(len(x)):
        i = int(np.floor(NCells*x[n]/LBox))
        j = int(np.floor(NCells*y[n]/LBox))
        if i in range(NCells) and j in range(NCells): 
            cells[i][j].add(n) # add particle index n to the set of particles in cell (i,j) #
            nSorted += 1
    if debug:
        print (nSorted, " of ", len(x)," particles sorted into the grid")
    return cells


def MultiParticleCollisionStep(NCells,m,pos,vel,xBox,yBox,x_pbc=False,y_pbc=False,debug=False):
    '''
        Returns unchanged positions, pos=(x,y), and modified velocities, vel=(vx,vy), 
        after multi-particle collisions in NCells x NCells cells

        m: particle masses

        xBox = (xmin,xmax) denotes the simulation box
        xBox = LBox corresponds to a box of (0,LBox)

        if x_pbc==True
            particles are sorted according to their folded positions,
            where the box position is randomly shifted to break Gallilein invariance
        else:
            particles inside the box are sorted according to their absolute positions
    
        the same rules apply in y-direction
    '''
    #Keep cells approximately square
    (xmin, xmax, XBox) = BoxDimensions(xBox)
    (ymin, ymax, YBox) = BoxDimensions(yBox)
    
    LBox = np.sqrt(XBox*YBox)
    NCellsX = np.int(np.round(NCells*XBox/LBox))
    NCellsY = np.int(np.round(NCells*YBox/LBox))
    
    NParticles = len(m)
    # sort particles into cells
    x = pos[0].copy()
    if x_pbc:
        x += XBox/NCellsX*(np.random.random()-0.5) # to break Gallilean invariance #
        x -= np.floor(x/XBox)*XBox # fold into box / minimum image convention for collisions #
    y = pos[1].copy() 
    if y_pbc:
        y += YBox/NCellsY*(np.random.random()-0.5) # to break Gallilean invariance #
        y -= np.floor(y/YBox)*YBox
    # cell numbers corresponding to x- and y-coordinates
    i = np.floor(NCellsX*x/XBox).astype(int)
    j = np.floor(NCellsY*y/YBox).astype(int)
    # combine into absolute cell number
    row = i + NCellsX*j
    if debug:
        print("row = ",row)
    # particle numbers
    col = np.array(range(NParticles))
    # combine into projector from particles to cells
    data = np.ones(NParticles)
    sparse_cell_projector = sparse.coo_matrix((data,(row,col)),shape=(NCellsX*NCellsY,NParticles))
    if debug:
        print("projector = ",sparse_cell_projector.todense())

    (vx,vy) = vel    
    # velocities of the CM of particles belonging to the same cell
    cell_vxCM = sparse_cell_projector.dot(m*vx)
    cell_vyCM = sparse_cell_projector.dot(m*vy)
    cell_M = sparse_cell_projector.dot(m)
    cell_M += (1-np.sign(cell_M))
    cell_vxCM /= cell_M
    cell_vyCM /= cell_M
    particle_vxCM = sparse.coo_matrix.transpose(sparse_cell_projector).dot(cell_vxCM)
    particle_vyCM = sparse.coo_matrix.transpose(sparse_cell_projector).dot(cell_vyCM)
    if debug:
        print("particle_vxCM = ",particle_vxCM)
# velocities of the cell CM frame
    particle_delta_vx = vx - particle_vxCM
    particle_delta_vy = vy - particle_vyCM
    if debug:
        print("particle_delta_vy = ",particle_delta_vy)
# Draw random rotation sense for each cell
    cell_rotation = 2*np.random.randint(2, size=NCellsX*NCellsY)-1
    particle_rotation = sparse.coo_matrix.transpose(sparse_cell_projector).dot(cell_rotation)
    if debug:
        print("particle_rotation = ",particle_rotation)
# New velocities: add randomly rotated relative velocity to cell CM velocity
    new_vx = particle_vxCM + particle_rotation*particle_delta_vy
    new_vy = particle_vyCM - particle_rotation*particle_delta_vx
    return pos, [new_vx,new_vy]


def IndividualWallCollision(m,x,vx,WallPosition,WallOrientation,dt):
    """
    Collisions with a reflecting wall positioned at WallPosition of size WallSize
    
    WallOrientation denotes the sign of the external normal vector 
    (i.e. for a Box [0,LBox] WallOrientation=-1 for the wall at x=0 and +1 for the wall at x=LBox)

    Returns 
        1-d array of position mirrored at the bounding wall
        1-d array of velocities, where the velocities of particles outside the box are reversed
        the total momentum transferred to the wall
    """

    IsReflected = 0.5*(np.sign(WallOrientation*(x-WallPosition))+1.0)

    new_x = (1-IsReflected)*x + IsReflected*(WallPosition - (x-WallPosition))
    new_vx = (1-2*IsReflected)*vx
    transferredMomentum = np.sum(2*m*vx*IsReflected)

    return (new_x, new_vx), transferredMomentum/dt

def WallPressureFromForcesOnWalls(f_Wall_xmax,f_Wall_xmin,f_Wall_ymax,f_Wall_ymin,XBox,YBox):
    """
    Returns the instantaneous wall pressure due to collisions
    """
    
    P = (f_Wall_xmax - f_Wall_xmin + f_Wall_ymax - f_Wall_ymin) / (2*XBox+2*YBox)
    return P

def WallCollisionStep(m,pos,vel,Box,dt,return_forces=False):
    """
    Collisions with the reflecting walls of the simulation cell [xBox,yBox]
    
    Returns 
        - (x,y) arrays of positions mirrored at the bounding walls
        - (vx,vy) arrays of velocities, where the velocities of particles outside the box are reversed
        - a pressure P estimate corresponding to the momentum transferred to the wall
          OR the forces f_Wall_xmin, f_Wall_xmax, f_Wall_ymin, f_Wall_ymaxm acting on the four walls
    """

    (xmin, xmax, XBox) = MD.BoxDimensions(Box[0])
    (ymin, ymax, YBox) = MD.BoxDimensions(Box[1])
    new_x = pos[0].copy()
    new_y = pos[1].copy()
    new_vx = vel[0].copy()
    new_vy = vel[1].copy()
    

    (new_x,new_vx), f_Wall_xmin = IndividualWallCollision(m,new_x,new_vx,xmin,-1,dt)
    (new_x,new_vx), f_Wall_xmax = IndividualWallCollision(m,new_x,new_vx,xmax,1,dt)

    (new_y,new_vy), f_Wall_ymin = IndividualWallCollision(m,new_y,new_vy,ymin,-1,dt)
    (new_y,new_vy), f_Wall_ymax = IndividualWallCollision(m,new_y,new_vy,ymax,1,dt)
    
#    P = (f_Wall_xmax - f_Wall_xmin + f_Wall_ymax - f_Wall_ymin) / (2*XBox+2*YBox)
    P = WallPressureFromForcesOnWalls(f_Wall_xmax,f_Wall_xmin,f_Wall_ymax,f_Wall_ymin,XBox,YBox)
    
    if return_forces:
        return (new_x,new_y), (new_vx,new_vy), f_Wall_xmin, f_Wall_xmax, f_Wall_ymin, f_Wall_ymax
    else:
        return (new_x,new_y), (new_vx,new_vy), P

############### Dynamics III: MD ########################################################################

def VelocityVerletTimeStepPartOne(m,pos,vel,force,delta_t):
    """
    The velocity Verlet time step needs to be executed in three steps:
    
        - First execute the present "Part One" to update the positions and velocity on the basis of the 
          forces corresponding to the positions at the beginning of the time steps
        - Next update the forces for the new positions at the end of the time step
        - Finish up by executing VelocityVerletTimeStepPartTwo to increment the velocities according to the
          forces corresponding to the positions at the end of the time steps
    """
    r = np.array(pos)
    v = np.array(vel)
    a = np.array(force)/m
    return r + delta_t * v + 0.5 * a * delta_t**2, v + 0.5 * delta_t * a


def VelocityVerletTimeStepPartTwo(m,pos,vel,force,delta_t):
    """
    The velocity Verlet time step needs to be executed in three steps:
    
        - First execute VelocityVerletTimeStepPartOne to update the positions and velocity on the basis of the 
          forces corresponding to the positions at the beginning of the time steps
        - Next update the forces for the new positions at the end of the time step
        - Finish up by executing the present VelocityVerletTimeStepPartTwo to increment the velocities according to the
          forces corresponding to the positions at the end of the time steps
    """
    r = np.array(pos)
    v = np.array(vel)
    a = np.array(force)/m
    return r, v + 0.5 * delta_t * a


################### Running simulations

def Generate_LJ_NVT_MolecularDynamics_Trajectory(d,m,LBox,kT,run_time,
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

    NParticles = m.size
    sigma = 1
    epsilon = 1
    #unit of time
    tau = sigma*np.sqrt(m[-1]/epsilon)      

    # define the length of the trajectory
    number_of_timesteps = int(np.round(run_time/time_step))

    #starting configuration
    if starting_configuration!=[]:
        [t,x,y,vx,vy] = starting_configuration
    else:
        # default initial state
        if start_from_stable_grid:
            x,y = StableGridPositionsIn2d(LBox,LBox,NParticles)
        else:
            x,y = GridPositionsIn2d(LBox,LBox,NParticles)
        vx = RandomVelocities(m,kT)
        vy = RandomVelocities(m,kT)
        t = 0
        if debug:
            print("No starting configuration")

    #initialize Trajectory
    t_tr = []
    x_tr = []
    vx_tr = []
    y_tr = []
    vy_tr = []

    fx,fy = LJ_forces_as_a_function_of_positions(d,epsilon,sigma,LBox,(x,y))
    # force for initial configuration needed for first time step

    for timestep in range(number_of_timesteps):
        (x,y),(vx,vy) = VelocityVerletTimeStepPartOne(m,(x,y),(vx,vy),(fx,fy),time_step)
        fx,fy = LJ_forces_as_a_function_of_positions(2,epsilon,sigma,LBox,(x,y))
        (x,y),(vx,vy) = VelocityVerletTimeStepPartTwo(m,(x,y),(vx,vy),(fx,fy),time_step)
        t += time_step
        
        t_tr.append(t)
        x_tr.append(x)
        vx_tr.append(vx)
        y_tr.append(y)
        vy_tr.append(vy)
    
        # thermostat: reinitialise velocities to control temperature
#        if np.mod( timestep*time_step, time_between_velocity_resets ) == 0.0 and timestep>1:
        if timestep%number_of_time_steps_between_velocity_resets == 0 and timestep>1:
            vx = RandomVelocities(m,kT)
            vy = RandomVelocities(m,kT)

    # convert trajectory lists to arrays to simplify the data analysis
    t_tr = np.array(t_tr)
    x_tr = np.array(x_tr)
    vx_tr = np.array(vx_tr)
    y_tr = np.array(y_tr)
    vy_tr = np.array(vy_tr)

    # analyse results 
    uPot_tr = LJ_energy_as_a_function_of_positions(d,epsilon,sigma,LBox,(x_tr,y_tr))
    uKin_tr = TotalKineticEnergy(m,vx_tr) + TotalKineticEnergy(m,vy_tr)
    pPot_tr = LJ_virial_pressure_as_a_function_of_positions(d,epsilon,sigma,LBox,(x_tr,y_tr)) 
    pKin_tr = KineticPressure_as_a_function_of_velocities(d,LBox,m,(vx_tr,vy_tr))
    pHyper_tr = LJ_hyper_virial_as_a_function_of_positions(d,epsilon,sigma,LBox,(x_tr,y_tr)) 
    
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



def Generate_Ensemble_of_LJ_NVT_MolecularDynamics_Trajectories(d,m,LBox,kT,NTrajectories,run_time,
                                                               list_of_starting_configurations=[],
                                                               time_step=0.01,
                                                               number_of_time_steps_between_stored_configurations=100,
                                                               number_of_time_steps_between_velocity_resets=100,
                                                               start_from_stable_grid=False,
                                                               debug=False):
    """
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
        ) = Generate_LJ_NVT_MolecularDynamics_Trajectory(d,m,LBox,kT,run_time,
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


def filterTheDict(dictObj, callback):
    '''
    Iterate over all the keys in dictionary and call the given callback function() on each key. 
    Add items for which callback() returns True to a new dictionary. 
    In the end return the new dictionary.
    '''
    newDict = dict()
    # Iterate over all the items in dictionary
    for key in dictObj.keys():
        # Check if item satisfies the given condition then add to new dict
        if callback(key):
            newDict[key] = dictObj[key]
    return newDict