# -*- coding: utf-8 -*-

# %% Preamble
import numpy as np
from . import RotationalKinematics as rot

# %% Standalones

# Expects 3x1 numpy matrix, produces 3x3 skew-symmetric numpy matrix
def tilde(vec):
    if not(isinstance(vec,np.matrix)):
        raise(Exception('Input to tilde must be a numpy matrix'))
    if np.shape(vec) != (3,1):
        raise(Exception('Shape of input to tilde must be (3,1)'))
    mat = np.matrix([
            [0        , -vec[2,0], vec[1,0] ],
            [vec[2,0] , 0        , -vec[0,0]],
            [-vec[1,0], vec[0,0] , 0        ]
            ])
    return mat

def unitvec(vec):
    if np.shape(vec)[1] != 1:
        raise(Exception('Shape of input to unitvec must be (n,1)'))
    u = vec / np.linalg.norm(vec)
    return u

# %% Inertia tensor class
# Initialized with I - 3x3 numpy matrix, an inertia tensor
class InertiaTensor:
    def __init__(self,I):
        # Error checking
        if not isinstance(I,np.matrix):
            raise(Exception('Assigned value to InertiaTensor must be a numpy matrix'))
        if np.shape(I) != (3,3):
            raise(Exception('Shape of assigned value to InertiaTensor must be shape (3,3)'))
        if (np.abs(I-I.T) > 1e-06).any():
            raise(Exception('Assigned inertia tensor not symmetric to within at least 1e-06'))
        # Re-symmetrize the tensor
        for i in range(0,3):
            for j in range(0,i):
                I[i,j] = I[j,i]
        self.I = I
    
    # Transform tensor from current frame (C) to new frame (F)
    # dcm_FC must be a DCM object (see RotationalKinematics)
    def frameTransformation(self,dcm_FC):
        # Error checking
        if not isinstance(dcm_FC,rot.DCM):
            raise(Exception('Provided DCM must be a DCM object (see RotationalKinematics)'))
        I_C = self.I
        I_F = dcm_FC.C.dot(I_C).dot(dcm_FC.T().C)
        return InertiaTensor(I_F)
    
    # Parallel axis
    # Transports tensor to describe inertia about a new point in the same frame
    # I_c - inertia about current point, expressed in current frame
    # I_p - inertia about new point, expressed in current frame
    # Rcp - 3x1 numpy matrix describing the vector to c from p in current frame
    # M   - scalar total mass
    def parallelAxis(self,M,Rcp):
        # Error checking
        if not isinstance(Rcp,np.matrix):
            raise(Exception('Provided Rcp to parallelAxis method of InertiaTensor must be a numpy matrix'))
        if np.shape(Rcp) != (3,1):
            raise(Exception('Provided Rcp to parallelAxis method of InertiaTensor must be shape 3x1'))
        if np.size(M) != 1:
            raise(Exception('Provided M to parallelAxis method of InertiaTensor must be size 1'))
        # Perform transport
        I_c = self.I
        I_p = I_c + M * tilde(Rcp).dot(tilde(Rcp).T)
        return InertiaTensor(I_p)
    
    # dot
    # interface to standard numpy dot
    def dot(self,a):
        return self.I.dot(a)
    
    # Obtain principle axis frame and transform to it
    # P - denotes principle axis frame
    # C - denotes current frame
    # Returns PC (dcm from C to P as a DCM object (see RotationalKinematics)) 
    #   and I_P (inertia tensor in principle axis frame)
    def principleAxis(self):
        I_C = self
        ev,evec = np.linalg.eig(I_C.I)
        # Ensure unit length of eigenvectors
        for i in range(0,3):
            evec[:,i] = unitvec(evec[:,i])
        # Ensure principle frame is right handed
        if np.cross(evec[:,0],evec[:,1],axis=0).T.dot(evec[:,2]) < 0:
            evec[:,2] = -evec[:,2]
            ev[2] = -ev[2]
        # Construct DCM
        PC = rot.DCM(evec.T)
        # Transform I_C to I_P
        I_P = I_C.frameTransformation(PC)
        return PC, I_P
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        