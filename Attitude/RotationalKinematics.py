# -*- coding: utf-8 -*-

# %% Preamble
import numpy as np
import math
import itertools as it


# %% Standalones    

# Expects 3x1 numpy matrix, produces 3x3 skew-symmetric numpy matrix
def tilde(vec):
    if not(isinstance(vec,np.matrix)):
        raise(Exception('Input to tilde must be a numpy matrix'))
    if np.shape(vec) != (3,1):
        raise(Exception('Shape of input to tilde must be (3,1)'))
    mat = np.matrix([
            [0.        , -vec[2,0], vec[1,0] ],
            [vec[2,0] , 0.        , -vec[0,0]],
            [-vec[1,0], vec[0,0] , 0.        ]
            ])
    return mat

def unitvec(vec):
    if np.shape(vec)[1] != 1:
        raise(Exception('Shape of input to unitvec must be (n,1)'))
    u = vec / np.linalg.norm(vec)
    return u


# %% DCM Class
    
# takes input of 3x3 numpy matrix
# exists to perform math relative to DCM transform matrices
class DCM:
    def __init__(self,C):
        if not(isinstance(C,np.matrix)):
            raise(Exception('Assigned value must be a numpy matrix'))
        if np.shape(C) != (3,3):
            raise(Exception('Shape of assigned DCM value must be (3,3)'))
        if np.sum( abs( C.T.dot(C) - np.eye(3) ) ) > 1e-3:
            raise(Exception('DCM is not orthonormal over all terms to a sum of at most 1e-3'))
        # reorthogonalize DCM via lapack's qr factorization
        Q, R = np.linalg.qr(C,mode='complete')
        for i in range(0,3):
            Q[:,i] = np.sign(R[i,i])*Q[:,i]
        self.C = C
        
    def inv(self):
        return DCM(self.C.T)
    
    T = inv
    
    # DCM_FN = DCM_FB.dot(DCM_BN)
    # DCM_FB = self
    # DCM_BN = C
    # DCM_FN = return
    def dot(self,C):
        if not(isinstance(C,DCM)):
            raise(Exception('Input to DCM dot method must be a DCM instance'))
        FB = self.C
        BN = C.C
        FN = FB.dot(BN)
        return DCM(FN)
    
    def quat(self):
        # utilizes Sheppard's method
        C = self.C
        Bmag =  np.sqrt(
                    np.matrix([
                        1/4*(1+np.trace(C)),
                        1/4*(1+2*C[0,0]-np.trace(C)),
                        1/4*(1+2*C[1,1]-np.trace(C)),
                        1/4*(1+2*C[2,2]-np.trace(C))
                    ])
                )
        Imax = np.argmax(Bmag)
        B = np.matrix(np.zeros([4,1]))
        B[Imax] = Bmag[0,Imax]
        if Imax==0:
            B[1,0] = (C[1,2]-C[2,1])/(4*Bmag[0,0])
            B[2,0] = (C[2,0]-C[0,2])/(4*Bmag[0,0])
            B[3,0] = (C[0,1]-C[1,0])/(4*Bmag[0,0])
        elif Imax==1:
            B[0,0] = (C[1,2]-C[2,1])/(4*Bmag[0,1])
            B[2,0] = (C[0,1]+C[1,0])/(4*Bmag[0,1])
            B[3,0] = (C[2,0]+C[0,2])/(4*Bmag[0,1])
        elif Imax==2:
            B[0,0] = (C[2,0]-C[0,2])/(4*Bmag[0,2])
            B[1,0] = (C[0,1]+C[1,0])/(4*Bmag[0,2])
            B[3,0] = (C[1,2]+C[2,1])/(4*Bmag[0,2])
        elif Imax==3:
            B[0,0] = (C[0,1]-C[1,0])/(4*Bmag[0,3])
            B[1,0] = (C[2,0]+C[0,2])/(4*Bmag[0,3])
            B[2,0] = (C[1,2]+C[2,1])/(4*Bmag[0,3])
        else:
            raise(Exception('This should be impossible.'))
        return Quaternion(B)
    
    ep = quat
    
    def prv(self):
        C = self.C
        phi  = math.acos(0.5*(C[0,0]+C[1,1]+C[2,2]-1)); # rad
        ehat = 1/(2*math.sin(phi))*np.matrix([[C[1,2]-C[2,1]],
                                             [C[2,0]-C[0,2]],
                                             [C[0,1]-C[1,0]]])
    
        return PRV(ehat, phi)
    
    def crp(self):
        C = self.C
        q = np.matrix(np.zeros((3,1)))
        
        zeta = np.sqrt( np.trace(C) +1 )
        q[0,0] = ( C[1,2] - C[2,1] ) / zeta**2
        q[1,0] = ( C[2,0] - C[0,2] ) / zeta**2
        q[2,0] = ( C[0,1] - C[1,0] ) / zeta**2
        return CRP(q)
    
    
    def mrp(self):
        C = self.C
        s = np.matrix(np.zeros((3,1)))
        zeta = np.sqrt( np.trace(self.C) +1 )
        if abs(zeta) > 1e-05:
            den  = zeta*(zeta+2)
            s[0,0] = ( C[1,2] - C[2,1] ) / den
            s[1,0] = ( C[2,0] - C[0,2] ) / den
            s[2,0] = ( C[0,1] - C[1,0] ) / den
            s = MRP(s)
        else:
            s = self.quat().mrp()
        return s
    
    # DCM_dot = f(DCM,w)
    # w is 3x1 numpy matrix
    # w is angular velocity of B frame with respect to N frame, expressed in B frame coordinates, in units of rad/sec
    def KDE(self,w):
        if not(isinstance(w,np.matrix)):
            raise(Exception('Angular velocity must be a numpy matrix'))
        if np.shape(w) != (3,1):
            raise(Exception('Shape of angular velocity must be (3,1)'))
        C = self.C
        wtilde = tilde(w)
        Cdot = -wtilde.dot(C)
        return Cdot
    
# %% Principle Rotation Vector Class
        
# Takes input of 3x1 numpy matrix (ehat) and scalar phi (radians)
class PRV:
    
    def __init__(self,ehat,phi):
        if abs(np.linalg.norm(ehat)-1.0) > 1e-06:
            raise(Exception('Principle Rotation Vector is not unit to within 1e-06'))
        self.ehat = ehat
        self.phi  = phi
        
    def dcm(self):
        ehat = self.ehat
        phi  = self.phi
        S = 1-math.cos(phi)
        C = np.matrix(np.zeros([3,3]))
        C[0,0] = ehat[0]**2*S+math.cos(phi)
        C[0,1] = ehat[0]*ehat[1]*S+ehat[2]*math.sin(phi)
        C[0,2] = ehat[0]*ehat[2]*S-ehat[1]*math.sin(phi)
        C[1,0] = ehat[1]*ehat[0]*S-ehat[2]*math.sin(phi)
        C[1,1] = ehat[1]**2*S+math.cos(phi)
        C[1,2] = ehat[1]*ehat[2]*S+ehat[0]*math.sin(phi)
        C[2,0] = ehat[2]*ehat[0]*S+ehat[1]*math.sin(phi)
        C[2,1] = ehat[2]*ehat[1]*S-ehat[0]*math.sin(phi)
        C[2,2] = ehat[2]**2*S+math.cos(phi)
        return DCM(C)
    
    # PRV_FN = PRV_FB.dot(PRV_BN)
    # PRV_FB = self
    # PRV_BN = prv
    # PRV_FN = return
    def dot(self,prv):
        phi_BN  = prv.phi
        ehat_BN = prv.ehat
        phi_FB  = self.phi
        ehat_FB = self.ehat
        
        phi  = 2*math.acos(math.cos(phi_BN/2)*math.cos(phi_FB/2)-
                           math.sin(phi_BN/2)*math.sin(phi_FB/2)*ehat_BN.T.dot(ehat_FB))
        ehat = (math.cos(phi_FB/2)*math.sin(phi_BN/2)*ehat_BN + 
                math.cos(phi_BN/2)*math.sin(phi_FB/2)*ehat_FB + 
                math.sin(phi_BN/2)*math.sin(phi_FB/2)*np.cross(ehat_BN,ehat_FB))
        ehat = ehat / math.sin(phi/2)
        return PRV(ehat,phi)
    
    
    def quat(self):
        phi = self.phi
        ehat = self.ehat
        B = np.matrix(np.zeros((4,1)))
        B[0,0] = math.cos(phi/2)
        B[1,0] = ehat[0,0]*math.sin(phi/2)
        B[2,0] = ehat[1,0]*math.sin(phi/2)
        B[3,0] = ehat[2,0]*math.sin(phi/2)
        return Quaternion(B)
    
    ep = quat
    
    def crp(self):
        phi  = self.phi
        ehat = self.ehat
        q = math.tan(phi/2) * ehat
        return CRP(q)
    
    def mrp(self):
        phi = self.phi
        ehat = self.ehat
        s = math.tan(phi/4) * ehat
        return MRP(s)
    
    # gamma_dot = f(gamma,w)
    # gamma is 3x1, phi*ehat
    # w is 3x1 numpy matrix
    # w is angular velocity of B frame with respect to N frame, expressed in B frame coordinates, in units of rad/sec
    def KDE(self,w):
        if not(isinstance(w,np.matrix)):
            raise(Exception('Angular velocity must be a numpy matrix'))
        if np.shape(w) != (3,1):
            raise(Exception('Shape of angular velocity must be (3,1)'))
        gamma = self.phi * self.ehat
        gammatilde = tilde(gamma)
        phi = self.phi
        gammadot = ( np.eye(3) + 0.5*gammatilde + phi**(-2)*(1-phi/(2*math.tan(phi/2))*gammatilde*gammatilde) ).dot(w)
        return gammadot
        

    
            
# %% Quaternion Class
        
# Takes input of 4x1 numpy matrix
# exists to execute quaternion math in DCM math format
# Must use the [svvv] formulation of quaternions
class Quaternion:
    
    def __init__(self,B):
        if not(isinstance(B,np.matrix)):
            raise(Exception('Assigned value must be a numpy matrix'))
        if np.shape(B) != (4,1):
            raise(Exception('Shape of assigned quaternion value must be (4,1)'))
        if np.abs(np.linalg.norm(B)-1) > 1e-3:
            raise(Exception('Quaternion is not of unit length to within 1e-3'))
        self.B = B / np.linalg.norm(B)
        
    def inv(self):
        tmp = np.matrix(np.zeros([4,1]))
        tmp[0] = self.B[0,0]
        tmp[1:4] = -self.B[1:4,0]
        return Quaternion(tmp)
    
    T = inv
    
    # Quat_FN = Quat_FB.dot(Quat_BN)
    # Quat_FB = self
    # Quat_BN = Quat
    # Quat_FN = return
    def dot(self,Quat):
        if not(isinstance(Quat,Quaternion)):
            raise(Exception('Input to Quaternion dot method must be a Quaternion instance'))
        B_FB = self.B
        B_BN = Quat.B
        B_FN = np.matrix([
                  [ B_FB[0,0],-B_FB[1,0],-B_FB[2,0],-B_FB[3,0]],
                  [ B_FB[1,0], B_FB[0,0], B_FB[3,0],-B_FB[2,0]],
                  [ B_FB[2,0],-B_FB[3,0], B_FB[0,0], B_FB[1,0]],
                  [ B_FB[3,0], B_FB[2,0],-B_FB[1,0], B_FB[0,0]]
                  ]).dot(B_BN)
        return Quaternion(B_FN)
    
    def dcm(self):
        B = np.squeeze(np.asarray(self.B))
        C = np.matrix(
                [[B[0]**2+B[1]**2-B[2]**2-B[3]**2,    2*(B[1]*B[2]+B[0]*B[3]),          2*(B[1]*B[3]-B[0]*B[2])        ],
                 [2*(B[1]*B[2]-B[0]*B[3]),            B[0]**2-B[1]**2+B[2]**2-B[3]**2,  2*(B[2]*B[3]+B[0]*B[1])        ],
                 [2*(B[1]*B[3]+B[0]*B[2]),            2*(B[2]*B[3]-B[0]*B[1]),          B[0]**2-B[1]**2-B[2]**2+B[3]**2]]
            )
        return DCM(C)
        
    # Return alternate form of Quaternion, same orientation but different path
    def alt(self):
        return Quaternion(-self.B)
    
    def crp(self):
        B = self.B
        q = B[1:4,0] / B[0,0]
        return CRP(q)
    
    def prv(self):
        return self.dcm().prv()
    
    def mrp(self):
        B = self.B
        s = B[1:4,0] / ( 1 + B[0,0] )
        return MRP(s)
    
    # Quat_dot = f(Quat,w)
    # w is 3x1 numpy matrix
    # w is angular velocity of B frame with respect to N frame, expressed in B frame coordinates, in units of rad/sec
    def KDE(self,w):
        if not(isinstance(w,np.matrix)):
            raise(Exception('Angular velocity must be a numpy matrix'))
        if np.shape(w) != (3,1):
            raise(Exception('Shape of angular velocity must be (3,1)'))
        B = self.B
        Bdot = 0.5 * np.matrix([
                [-B[1,0],-B[2,0],-B[3,0]],
                [ B[0,0],-B[3,0], B[2,0]],
                [ B[3,0], B[0,0],-B[1,0]],
                [-B[2,0], B[1,0], B[0,0]]
                ]).dot(w)
        return Bdot
        
        
    
    
# %% Euler Parameter Class (alias for Quaternion class)
EP = Quaternion
    
    
# %% Classical Rodriguez Parameter Class
        
# Takes input of 3x1 numpy matrix
# Exists to execute classical rodriguez parameter math in DCM math format
class CRP:
    
    def __init__(self,q):
        if not(isinstance(q,np.matrix)):
            raise(Exception('Assigned value must be a numpy matrix'))
        if np.shape(q) != (3,1):
            raise(Exception('Shape of assigned CRP value must be (3,1)'))
        self.q = q
    
    def inv(self):
        return CRP(-self.q)
    
    T = inv
    
    # CRP_FN = CRP_FB.dot(CRP_BN)
    # CRP_FB = self
    # CRP_BN = crp
    # CRP_FN = return
    def dot(self,crp):
        if not(isinstance(crp,CRP)):
            raise(Exception('Input to CRP dot method must be a CRP instance'))
        q_FB = self.q
        q_BN = crp.q
        q_FN = np.asscalar(1/(1-q_FB.T.dot(q_BN))) * (q_FB + q_BN - np.cross(q_FB,q_BN,axis=0))
        return CRP(q_FN)
    
    def dcm(self):
        q = np.squeeze(np.asarray(self.q))
        C = np.matrix([
                [1+q[0]**2-q[1]**2-q[2]**2, 2*(q[0]*q[1]+q[2]),        2*(q[0]*q[2]-q[1])       ],
                [2*(q[1]*q[0]-q[2]),        1-q[0]**2+q[1]**2-q[2]**2, 2*(q[1]*q[2]+q[0])       ],
                [2*(q[2]*q[0]+q[1]),        2*(q[2]*q[1]-q[0]),        1-q[0]**2-q[1]**2+q[2]**2]
                ]) / (1 + self.q.T.dot(self.q))
        return DCM(C)
    
    def prv(self):
        return self.dcm().prv()
    
    def quat(self):
        B = np.matrix(np.zeros((4,1)))
        q = self.q
        den = np.sqrt( 1 + q.T.dot(q) )
        B[0,0] = 1 / den
        B[1:4,0] = q / den
        return Quaternion(B)
    
    ep = quat
    
    def mrp(self):
        q = self.q
        den = np.asscalar( 1 + np.sqrt(1+q.T.dot(q)) )
        s = q / den
        return MRP(s)
    
    # CRP_dot = f(CRP,w)
    # w is 3x1 numpy matrix
    # w is angular velocity of B frame with respect to N frame, expressed in B frame coordinates, in units of rad/sec
    def KDE(self,w):
        if not(isinstance(w,np.matrix)):
            raise(Exception('Angular velocity must be a numpy matrix'))
        if np.shape(w) != (3,1):
            raise(Exception('Shape of angular velocity must be (3,1)'))
        q = self.q
        qdot = 0.5*np.matrix([
                [1+q[0,0]**2         , q[0,0]*q[1,0]-q[2,0], q[0,0]*q[2,0]+q[1,0]],
                [q[1,0]*q[0,0]+q[2,0], 1+q[1,0]**2         , q[1,0]*q[2,0]-q[0,0]],
                [q[2,0]*q[0,0]-q[1,0], q[2,0]*q[1,0]+q[0,0], 1+q[2,0]**2     ]
                ]).dot(w)
        return qdot
    
    
    
    
# %% Modified Rodriguez Parameter Class
        
# Takes input of 3x1 numpy matrix
# Exists to execute modified rodriguez parameter math in DCM math format
class MRP:
    
    def __init__(self,s):
        if not(isinstance(s,np.matrix)):
            raise(Exception('Assigned value must be a numpy matrix'))
        if np.shape(s) != (3,1):
            raise(Exception('Shape of assigned MRP value must be (3,1)'))
        self.s = s
        
    def inv(self):
        return MRP(-self.s)
    
    T = inv
    
    # return shadow set
    def shadow(self):
        s = self.s
        return MRP( -s / ( s.T.dot(s) ) )
    
    # return short rotation mrp
    def shortRotation(self):
        s = self
        if np.linalg.norm(self.s) > 1:
            s = self.shadow()
        return s
    
    # MRP_FN = MRP_FB.dot(MRP_BN)
    # MRP_FB = self
    # MRP_BN = mrp
    # MRP_FN = return
    def dot(self,mrp):
        if not(isinstance(mrp,MRP)):
            raise(Exception('Input to MRP dot method must be an MRP instance'))
        s_FB = self.s
        s_BN = mrp.s
        den = 1 + s_FB.T.dot(s_FB) * s_BN.T.dot(s_BN) - 2*s_BN.T.dot(s_FB)
        # Avoid singularity
        if abs(den) < 1e-05:
            s_FB = s_FB.shadow()
            den = 1 + s_FB.T.dot(s_FB) * s_BN.T.dot(s_BN) - 2*s_BN.T.dot(s_FB)
        s_FN = (np.asscalar((1-s_BN.T.dot(s_BN)))*s_FB +
                np.asscalar((1-s_FB.T.dot(s_FB)))*s_BN -
                2*np.cross(s_FB,s_BN,axis=0))
        s_FN = s_FN / np.asscalar(den)
        return MRP(s_FN)
    
    def dcm(self):
        s = self.s
        smag = np.asscalar(np.linalg.norm(s))
        stilde = tilde(s)
        C = np.eye(3) + ( 8*stilde.dot(stilde) - 4*(1-smag**2)*stilde ) / (1+smag**2)**2
        return DCM(C)
    
    def crp(self):
        s = self.s
        den = np.asscalar( 1 - s.T.dot(s) )
        q = 2 * s / den
        return CRP(q)
    
    def quat(self):
        s = self.s
        smag = np.asscalar(np.linalg.norm(s))
        B = np.matrix(np.zeros((4,1)))
        B[0,0] = ( 1 - smag**2 ) / ( 1 + smag**2 )
        B[1:4,0] = 2 * s / (1 + smag**2 )
        return Quaternion(B)
    
    def prv(self):
        return self.dcm().prv()
    
    # MRP_dot = f(MRP,w)
    # w is 3x1 numpy matrix
    # w is angular velocity of B frame with respect to N frame, expressed in B frame coordinates, in units of rad/sec
    def KDE(self,w):
        if not(isinstance(w,np.matrix)):
            raise(Exception('Angular velocity must be a numpy matrix'))
        if np.shape(w) != (3,1):
            raise(Exception('Shape of angular velocity must be (3,1)'))
        s = self.s
        stilde = tilde(s)
        smag   = np.asscalar(np.linalg.norm(s))
        sdot = 0.25*( (1-smag**2)*np.eye(3) + 2*stilde + 2*s.dot(s.T) ).dot(w)
        return sdot
    
        
# %% Euler Angle Class
# ang MUST BE in radians
# ang - 3x1 numpy matrix of first,second,third rotation angle, radians
# order - tuple of rotation order, e.g. (3,2,1), (3,1,3)
class EulerAngle:
    def __init__(self,ang,order):
        # Error checking
        if not(isinstance(ang,np.matrix)):
            raise(Exception('Assigned ang (angles) value must be a numpy matrix'))
        if np.shape(ang) != (3,1):
            raise(Exception('Shape of assigned EulerAngle ang (angles) value must be (3,1)'))
        if not(isinstance(order,tuple)):
            raise(Exception('Assigned order must be a tuple e.g. (3,2,1), (3,1,3),etc'))
        l = list(it.permutations(range(1,4)))
        l.extend( ( (1,2,1),(1,3,1),(2,1,2),(2,3,2),(3,1,3),(3,2,3) ) )
        if order not in l:
            raise(Exception('Assigned order must be a valid euler angle orer, e.g. (3,2,1), (3,1,3),etc.'))
        self.ang = ang
        self.order = order
        
    # Elementary matrices
    def _M(self,theta,axis): # theta in rad
        if axis not in (1,2,3):
            raise(Exception('Axis argument to M method of EulerAngle must be either 1, 2, or 3'))
        if axis == 1:
            return np.matrix([[1.,0,0],[0,math.cos(theta),math.sin(theta)],[0,-math.sin(theta),math.cos(theta)]])
        elif axis == 2:
            return np.matrix([[math.cos(theta),0.,-math.sin(theta)],[0,1,0],[math.sin(theta),0,math.cos(theta)]])
        elif axis == 3:
            return np.matrix([[math.cos(theta),math.sin(theta),0.],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        else:
            raise(Exception('Oops, this shouldn''t be possible'))
        
    def dcm(self):
        ang = self.ang
        order = self.order
        Mfirst  = self._M(ang[0],order[0])
        Msecond = self._M(ang[1],order[1])
        Mthird  = self._M(ang[2],order[2])
        return DCM(Mthird.dot(Msecond).dot(Mfirst))
            
    







