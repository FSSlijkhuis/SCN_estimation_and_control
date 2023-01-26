import numpy as np
import control

def FE_init(time,dt):
    """Initialization of Forward Euler.
    Given simulation time (seconds) and timestep size (dt), returns list of times (for looping purposes) and the number of total timesteps (Nt).

    Args:
        time (integer): the number of seconds to run the FE simulation for, in seconds.
        dt (float): the length of a single timestep, in seconds.

    Returns:
        ndarray: array of times to loop over, or for plotting purposes.
        int: the total number of timesteps in the simulation.
    """    
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
Initialization of Cartpole system.
Given cartpole parameters, returns A and B matrices for cartpole in state-space form.
"""
def Cartpole_init(m = 1,M = 5,L = 2,g = -10,d = 1,s = 1):

    #A-matrix, defines dynamics of the DS
    A = np.array([[0,1,0,0],
              [0,(-d/M),(-m*g/M),0],
              [0,0,0,1],
              [0,(-s*d/(M*L)),(-s*(m+M)*g/(M*L)),0]])

    #B-matrix, defines the influence of the force on the system
    B = np.array([[0],[(1/M)],[0],[(s*1/(M*L))]])

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
    Kf_t,_,_=control.lqr(np.transpose(A),np.transpose(C),Vd,Vn)
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
def KfSCN_init(K,Nt,A,B,C,Kf,N=100,lam=0.1,bounding_box_factor=10,zero_init=True,x0=None,seed=0):

    np.random.seed(seed)

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
        r[:,0] = np.array(np.linalg.pinv(D)@x0) # pseudo-inverse - "cheaty" way of getting the right firing rate
        V[:,0] = D.T@(x0-D@r[:,0])

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
Initialization of SCN Controller.
Given the SCN and FE parameters, returns SCN states with connections.
"""
def ControllerSCN_init(K,Nt,A,B,C,Kf,Kc,N=100,lam=0.1,bounding_box_factor=10,zero_init=True,x0=None,seed=0):

    np.random.seed(seed)

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
        r[:,0] = np.array(np.linalg.pinv(D)@x0) # pseudo-inverse - "cheaty" way of getting the right firing rate
        V[:,0] = D.T@(x0-D@r[:,0])

    #We require an index for the weights, as the connections are only relevant for the first K/2 weights (the rest are for encoding the target state)
    i=int(K/2)

    # Network connections:
    # - fast
    O_f = D[:-i].T @ D[:-i]
    # - slow
    O_s = D[:-i].T @ (lam*np.identity(i) + A) @ D[:-i]
    # - rec. control
    O_c = -D[:-i].T @ B @ Kc @ D[:-i]
    # - ff. control
    F_c = D[:-i].T @ B @ Kc
    # - rec. kalman
    O_k = D[:-i].T @ Kf @ C @ D[:-i]
    # - ff kalman
    F_k = -D[:-i].T @ Kf

    return D,T,V,s,r,O_f,O_s,O_c,F_c,O_k,F_k

"""
Initialization of the Kf loop.
Given parameters, returns other matrices which are of importance in the estimation loop, such as the observation matrix Y, noise matrices, control matrix U. 
Also returns X_hat and X_hat_fe, the matrices which keep track of the Kf estimation for both the SCN and the idealized KF respectively.
"""
def KfLoop_init(X,A,B,C,x0,Nt,Vd_cov,Vn_cov):

    U = np.zeros([B.shape[1],Nt+1])
    Y = np.zeros([C.shape[0],Nt+1])
    X_hat = np.zeros([len(x0),Nt+1])
    X_hat_fe = np.zeros([len(x0),Nt+1])
    
    Vd = Vd_cov*np.identity(len(A))
    Vn = Vn_cov*np.identity(len(A))

    uDIST = np.random.multivariate_normal(np.zeros(len(A)),Vd,Nt+1).T
    uNOISE = np.random.multivariate_normal(np.zeros(len(A)),Vn,Nt+1).T

    Y[:,0] = C@X[:,0] + uNOISE[:,0]

    return U,Y,X_hat,X_hat_fe,uDIST,uNOISE

"""
Initialization of the Control loop.
Given parameters, returns other matrices which are of importance in the control loop, such as the observation matrix Y, noise matrices, control matrix U. 
Also returns X_hat and X_hat_fe, the matrices which keep track of the controller estimation for both the SCN and the idealized controller respectively.
"""
def ControlLoop_init(X,X_2,error_scn,error_ideal,x_des,dt,A,B,C,x0,Nt,Vd_cov,Vn_cov):

    U = np.zeros([B.shape[1],Nt+1])
    Y = np.zeros([C.shape[0],Nt+1])
    
    U_2 = np.zeros([B.shape[1],Nt+1])
    Y_2 = np.zeros([C.shape[0],Nt+1])
    
    X_hat = np.zeros([len(x0),Nt+1])
    X_hat_fe = np.zeros([len(x0),Nt+1])
    
    Vd = Vd_cov*np.identity(len(A))
    Vn = Vn_cov*np.identity(len(A))

    uDIST = np.random.multivariate_normal(np.zeros(len(A)),Vd,Nt+1).T
    uNOISE = np.random.multivariate_normal(np.zeros(len(A)),Vn,Nt+1).T

    Y[:,0] = C@X[:,0] + uNOISE[:,0]
    Y_2[:,0] = C@X_2[:,0] + uNOISE[:,0]
    
    Dx=np.gradient(x_des,axis=1)/dt
    
    error_scn[:,0] = np.abs(X[:,0]-x_des[:,0])
    error_ideal[:,0] = np.abs(X_2[:,0]-x_des[:,0])

    return U,Y,U_2,Y_2,X_hat,X_hat_fe,uDIST,uNOISE,Dx,error_scn,error_ideal