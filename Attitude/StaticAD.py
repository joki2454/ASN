# -*- coding: utf-8 -*-

# %% Preamble
import numpy as np
from . import RotationalKinematics as rot
from scipy import optimize

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

# %% Triad Method
# NOTE: here, m1 is by default the more trusted vector
# m1_B - 3x1 numpy matrix, measurement 1 unit vector, expressed in body frame
# m2_B - 3x1 numpy matrix, measurement 2 unit vector, expressed in body frame
# m1_N - 3x1 numpy matrix, measurement 1 unit vector, expressed in inertial frame
# m2_N - 3x1 numpy matrix, measurement 2 unit vector, expressed in inertial frame 
# 
# Returns BN as a DCM object (see RotationalKinematics)
def triad(m1_B,m2_B,m1_N,m2_N):
    # Error checking
    collect = (m1_B,m2_B,m1_N,m2_N)
    for i in range(0,np.shape(collect)[0]):
        if not(isinstance(collect[i],np.matrix)):
            raise(Exception('All inputs to triad method of StaticAD must be of type np.matrix'))
        if np.shape(collect[i]) != (3,1):
            raise(Exception('All inputs to triad method of StaticAD must be of shape (3,1)'))
        if abs(np.linalg.norm(collect[i])-1) > 1e-03:
            raise(Exception('At least one input to triad is not of unit length to within 1e-03'))
    # Re-unitize
    m1_B = unitvec(m1_B)
    m2_B = unitvec(m2_B)
    m1_N = unitvec(m1_N)
    m2_N = unitvec(m2_N)
    # Construct T frame expressed in B frame
    t1_B = m1_B
    t2_B = unitvec( np.cross(m1_B,m2_B,axis=0) )
    t3_B = np.cross(t1_B,t2_B,axis=0)
    # Construct T frame expressed in N frame
    t1_N = m1_N
    t2_N = unitvec( np.cross(m1_N,m2_N,axis=0) )
    t3_N = np.cross(t1_N,t2_N,axis=0)
    # Construct BT and NT DCMs as DCM objects
    BT = rot.DCM(np.matrix(np.concatenate((t1_B,t2_B,t3_B),axis=1)))
    NT = rot.DCM(np.matrix(np.concatenate((t1_N,t2_N,t3_N),axis=1)))
    # Construct BN DCM
    BN = BT.dot(NT.T())
    return BN

# %% Devenport's q-method
# NOTE: This method does not check that inputs are unit vectors, it simply unitizes them
# vB - 3xn numpy array or matrix of all (column vector) measurements in the body     frame
# vN - 3xn numpy array or matrix of all (column vector) measurements in the inertial frame
# weights  - 1xn numpy array or matrix of weights corresponding to each measurement
# Returns BN as a Quaternion object (see RotationalKinematics)
def devenport_qmethod(vB,vN,weights):
    #Error Checking
    if (np.shape(vB)[0] != 3) or (np.shape(vN)[0] != 3):
        raise(Exception('vB and vN inputs to devenport_qmethod method of StaticAD must be of shape (3,n)'))
    if np.shape(vB)[1] != np.shape(vN)[1]:
        raise(Exception('vB and vN inputs to devenport_qmethod method of StaticAD must have same number of columns/measurement vectors'))
    if np.shape(weights) != (1,np.shape(vN)[1]):
        raise(Exception('w input to devenport_qmethod method of StaticAD must must be the same length as the number of columns/measurement vectors'))
    # Unitize all measurements
    for i in range(0,np.shape(vB)[1]):
        vB[:,i] = unitvec(vB[:,i])
        vN[:,i] = unitvec(vN[:,i])
    # Construct K matrix
    B = np.matrix(np.zeros((3,3)))
    for i in range(0,np.shape(vB)[1]):
        B = B + weights[0,i]*vB[:,i].dot(vN[:,i].T)
    S = B + B.T
    sigma = np.matrix(np.trace(B))
    Z = np.matrix([
            [B[1,2]-B[2,1]],
            [B[2,0]-B[0,2]],
            [B[0,1]-B[1,0]]
            ])
    Ktop = np.concatenate((sigma,Z.T),axis=1)
    Kbot = np.concatenate((Z,S-np.asscalar(sigma)*np.eye(3)),axis=1)
    K = np.concatenate((Ktop,Kbot),axis=0)
    # Obtain eigenvalues and eigenvectors of K matrix
    (D,V) = np.linalg.eig(K)
    # Extract eigenvector corresponding to max eigenvalue, and return as quaternion object
    Imax = np.argmax(D)
    B_BN = np.matrix(unitvec(V[:,Imax]))
    return rot.EP(B_BN)
    
# %% QUEST
# NOTE: This method does not assume that inputs are unit vectors, it simply unitizes them
# vB - 3xn numpy array or matrix of all (column vector) measurements in the body     frame
# vN - 3xn numpy array or matrix of all (column vector) measurements in the inertial frame
# weights  - 1xn numpy array or matrix of weights corresponding to each measurement
# Returns BN as a Quaternion object (see RotationalKinematics)
def QUEST(vB,vN,weights):
    #Error Checking
    if (np.shape(vB)[0] != 3) or (np.shape(vN)[0] != 3):
        raise(Exception('vB and vN inputs to devenport_qmethod method of StaticAD must be of shape (3,n)'))
    if np.shape(vB)[1] != np.shape(vN)[1]:
        raise(Exception('vB and vN inputs to devenport_qmethod method of StaticAD must have same number of columns/measurement vectors'))
    if np.shape(weights) != (1,np.shape(vN)[1]):
        raise(Exception('w input to devenport_qmethod method of StaticAD must must be the same length as the number of columns/measurement vectors'))
    # Unitize all measurements
    for i in range(0,np.shape(vB)[1]):
        vB[:,i] = unitvec(vB[:,i])
        vN[:,i] = unitvec(vN[:,i])
    # Construct K matrix
    B = np.matrix(np.zeros((3,3)))
    for i in range(0,np.shape(vB)[1]):
        B = B + weights[0,i]*vB[:,i].dot(vN[:,i].T)
    S = B + B.T
    sigma = np.matrix(np.trace(B))
    Z = np.matrix([
            [B[1,2]-B[2,1]],
            [B[2,0]-B[0,2]],
            [B[0,1]-B[1,0]]
            ])
    Ktop = np.concatenate((sigma,Z.T),axis=1)
    Kbot = np.concatenate((Z,S-np.asscalar(sigma)*np.eye(3)),axis=1)
    K = np.concatenate((Ktop,Kbot),axis=0)
    # Take sum of weights to be initial guess for max eigenvalue
    ev0 = np.sum(weights)
    # Iterate using characteristic equation of K matrix to find max eigenvalue
    def f(s):
        return np.linalg.det(K-s*np.eye(4))
    P = np.poly(K)
    def dfds(s):
        return 4*P[0]*s**3 + 3*P[1]*s**2 + 2*P[2]*s + P[3]
    ev = optimize.newton(f,ev0,fprime=dfds)
    # Calculate the CRP
    A = np.matrix((ev+np.asscalar(sigma))*np.eye(3) - S)
    crp = rot.CRP(np.linalg.solve(A,Z))
    B_BN   = crp.quat()
    return B_BN

# %% OLAE
# NOTE: This method does not assume that inputs are unit vectors, it simply unitizes them
# vB - 3xn numpy array or matrix of all (column vector) measurements in the body     frame
# vN - 3xn numpy array or matrix of all (column vector) measurements in the inertial frame
# weights  - 1xn numpy array or matrix of weights corresponding to each measurement
# Returns BN as a Quaternion object (see RotationalKinematics)
def OLAE(vB,vN,weights):
    #Error Checking
    if (np.shape(vB)[0] != 3) or (np.shape(vN)[0] != 3):
        raise(Exception('vB and vN inputs to devenport_qmethod method of StaticAD must be of shape (3,n)'))
    if np.shape(vB)[1] != np.shape(vN)[1]:
        raise(Exception('vB and vN inputs to devenport_qmethod method of StaticAD must have same number of columns/measurement vectors'))
    if np.shape(weights) != (1,np.shape(vN)[1]):
        raise(Exception('w input to devenport_qmethod method of StaticAD must must be the same length as the number of columns/measurement vectors'))
    # Unitize all measurements
    for i in range(0,np.shape(vB)[1]):
        vB[:,i] = unitvec(vB[:,i])
        vN[:,i] = unitvec(vN[:,i])
    # Construct D vector and S,W matrices
    N = np.shape(vB)[1]
    D = np.matrix(np.zeros((3*N,1)))
    S = np.matrix(np.zeros((3*N,3)))
    W = np.matrix(np.zeros((3*N,3*N)))
    for i in range(0,N):
        D[3*i:3*(i+1),0]   = vB[:,i]-vN[:,i]
        S[3*i:3*(i+1),0:3] = tilde(vB[:,i]+vN[:,i])
        for j in range(0,3):
            W[i+j,i+j] = weights[0,i]
    q_BN = np.linalg.solve(S.T.dot(W).dot(S),S.T.dot(W).dot(D))
    return rot.CRP(q_BN)
    
    




