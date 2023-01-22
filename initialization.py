import numpy as np
import control

"""
Initialization of Forward Euler.
Given simulation time (seconds) and timestep size (dt), returns list of times (for looping purposes) and the number of total timesteps (Nt).
"""
def FE_init(time,dt):

    #Forward Euler parameters
    times = np.arange(0, time, dt)
    Nt=len(times)

    return times,Nt

"""
Initialization of SMD system.
Given m, k and c (SMD parameters), returns A and B matrices for SMD system in state-space form.
"""
def SMD_init(m=3,k=5,c=0.5):

    #A-matrix, defines dynamics of the DS
    A = np.array([[0,1],
            [(-k/m),(-c/m)]]) 

    #B-matrix, defines the influence of the force on the system
    B = np.array([[0],[1/m]]) 

    #Check controllability 
    print("Rank of controllability-matrix:", np.linalg.matrix_rank(control.ctrb(A,B)))

    return A,B

"""
Initialization of LQR gain matrix.
Given A and B matrices, as well as LQR parameters Q and R, returns LQR gain matrix Kc. 
"""
def Control_init(A,B,Q,R):

    #LQR gain matrix calculation
    Kc,_,_ = control.lqr(A,B,Q,R)

    return Kc

"""
Initialization of Kalman filter gain matrix.
Given A and C matrices, as well as covariance parameters for disturbance and noise, returns Kalman filter gain matrix Kf.
"""
def Kalman_init(A,C,Vn_cov=0.001,Vd_cov=0.001):

    #Covariance matrices
    Vd = Vd_cov*np.identity(len(A))  # disturbance covariance
    Vn = Vn_cov*np.identity(len(A))    # noise covariance

    #Kalman filter gain matrix calculation
    Kf_t,_,_=control.lqr(np.transpose(A),np.transpose(C),Vn,Vd)
    Kf=np.transpose(Kf_t)
    return Kf

"""
Initialization of state matrix X.
Given the starting state x0 and FE parameter Nt, returns X, a zero-matrix which keeps track of the system state.
"""
def X_init(x0,Nt):
    #Initialization of 'real system'
    X=np.zeros([len(x0),Nt+1])
    X[:,0]=x0

    return X

"""
Initialization of Kalman Filter SCN.
Given the SCN and FE parameters, returns SCN states with connections.
"""
def KfSCN_init(K,Nt,A,B,C,Kf,N=100,lam=0.1,bounding_box_factor=10,zero_init=True):

    D=np.random.randn(K,N) # N x K - Weights associated to each neuron
    D=D/np.linalg.norm(D,axis=0) #normalize
    D = D / bounding_box_factor # avoid too big discontinuities
    T = np.diag(D.T@D)/2

    # Initialize Voltage, spikes, rate
    V = np.zeros([N,Nt+1])
    s = np.zeros([N,Nt+1])
    r = np.zeros([N,Nt+1])

    # Set initial conditions
    if not zero_init:
        r[:,0] = np.array(np.linalg.pinv(D)@np.array([-5, 0])) # pseudo-inverse - "cheaty" way of getting the right firing rate
        V[:,0] = D.T@(np.array([-5, 0])-D@r[:,0])

    # Network connections:
    # - fast
    O_f = D.T @ D
    # - slow
    O_s = D.T @ (lam*np.identity(K) + A) @ D
    # - external input
    F_i = D.T @ B
    # - rec. kalman
    O_k = -D.T @ Kf @ C @ D
    # - ff kalman
    F_k = D.T @ Kf

    return D,T,V,s,r,O_f,O_s,F_i,O_k,F_k

"""
Initialization of the Kf loop.
Given parameters, returns other matrices which are of importance in the control loop, such as the observation matrix Y, noise matrices, control matrix U. 
Also returns X_hat and X_hat_fe, the matrices which keep track of the Kf estimation for both the SCN and the idealized KF respectively.
"""
def KfLoop_init(X,B,C,x0,Nt,Vd,Vn):

    U = np.zeros([B.shape[1],Nt+1])
    Y = np.zeros([C.shape[0],Nt+1])
    X_hat = np.zeros([len(x0),Nt+1])
    X_hat_fe = np.zeros([len(x0),Nt+1])

    uDIST = np.random.multivariate_normal([0,0],Vd,Nt+1).T
    uNOISE = np.random.multivariate_normal([0,0],Vn,Nt+1).T

    Y[:,0] = C@X[:,0] + uNOISE[:,0]

    return U,Y,X_hat,X_hat_fe,uDIST,uNOISE