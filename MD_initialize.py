import numpy as np
import array
################################### INITIALIZATION ###################################################

class md_settings():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.N = self.NumberOfBoxesPerDimension**3 * 4
        self.L = (self.N/self.rho)**(1.0/3.0)
        self.T = self.TSI/119.8
        self.TruncR = self.L/np.sqrt(2) * 1.0001
    def update(**kwargs):
        self.__dict__.update(kwargs)

class md_instance():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(**kwargs):
        self.__dict__.update(kwargs)

def initialise(settings, seed):
    np.random.seed(seed)

    r = InitialisePositions(settings)
    v = InitialiseVelocities(settings)

    return r, v
    

def InitialisePositions (settings):
    NumberOfBoxesPerDimension = settings.NumberOfBoxesPerDimension
    L = settings.L
    NumberOfTimeSteps = settings.NumberOfTimeSteps
    N = settings.N

    a = L/NumberOfBoxesPerDimension
    r = np.zeros((3,N,NumberOfTimeSteps+1))
    EpsilonBoundary = 0.0001 * a #We don't want to put the particles exactly on the boundary, because it won't be clear
    #wether they belong inside the volume or the next volume. 
    GridVector = np.linspace(0,L-a,NumberOfBoxesPerDimension) #A vector used to build the fcc lattice. 
    (x,y,z) = np.meshgrid(GridVector,GridVector,GridVector)
    rCubenodes = (np.vstack((x.flatten(1),y.flatten(1),z.flatten(1)))).transpose()
    rShiftedInxAndy = rCubenodes + np.tile(np.array([0.5*a,0.5*a,0]),(np.size(x),1))
    rShiftedInxAndZ = rCubenodes + np.tile(np.array([0.5*a,0,0.5*a]),(np.size(x),1))
    rShiftedInyAndZ = rCubenodes + np.tile(np.array([0,0.5*a,0.5*a]),(np.size(x),1))
    rTemp = np.vstack((rCubenodes,rShiftedInxAndy,rShiftedInxAndZ,rShiftedInyAndZ)) #r is a matrix containing the positions
    #of the particles initially configured in the fcc lattice.

    rTemp += EpsilonBoundary #A small offset is added to the position of every particle to ensure that none of them are
    #located on the boundary.
    return rTemp.T

def InitialiseVelocities (settings):
    sigma = np.sqrt(settings.T)
    mu = 0
    v = np.zeros((3,settings.N,settings.NumberOfTimeSteps+1))
    VelocityGenerated = sigma * np.random.randn(3,settings.N) + mu
    AverageVelocity = np.mean(VelocityGenerated,1)
    return VelocityGenerated - AverageVelocity[:,np.newaxis]

#####################################################################
settings = md_settings(TSI=119.8,
                          rho=1.0,
                          NumberOfBoxesPerDimension=12,
                          NumberOfTimeSteps=2000,
                          TruncR=3000,
                          h=0.004)
# Initialise the system
r = InitialisePositions(settings)
v = InitialiseVelocities(settings)

fid_r = open('r_init','wb')
r_init = array.array('f',(r.T).flatten())
for i in range(settings.N):
	print(r_init[(3*i):(3*i+3)],i)
r_init.tofile(fid_r)
fid_r.close()

fid_v = open('v_init','wb')
v_init = array.array('f',(v.T).flatten())
v_init.tofile(fid_v)
fid_v.close()
print(settings.L)

