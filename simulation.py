import numpy as np

"""
Function for running a single step of the SCN Kalman filter network.
"""
def run_KfSCN_step(y,u,r,s,v,D,T,lam,O_f,O_s,F_i,O_k,F_k,C,t,dt,sigma):

    # Calculating the voltages at time t+1
    dvdt = -lam * v - O_f @ s + O_s @ r + F_i @ u + (O_k @ r + F_k @ C @ y)
    v_next = v + dvdt*dt + np.sqrt(dt)*sigma*np.random.randn(len(dvdt))

    # check if there are neurons whose voltage is above threshold
    above = np.where(v_next > T)[0]

    # introduce a control to let only one neuron fire at the time
    s_next=np.zeros(s.shape)
    if len(above):
        s_next[np.argmax(v_next)] = 1/dt

    # update rate
    drdt = s_next - lam*r
    r_next = r + drdt*dt
    
    return r_next, s_next, v_next

"""
Function for running a single step of the SCN controller.
"""
def run_SCNcontrol_step(y,x_des,Dx,r,s,v,D,T,lam,Kc,O_f,O_s,O_c,F_c,O_k,F_k,B,C,t,dt,sigma):
    
    #We require an index for the weights, as the connections are only relevant for the first B weights (the rest are for encoding the target state)
    i=len(B)
    
    u_next = -Kc @ (D[:-i] @ r - D[i:] @ r)

    # Calculating the voltages at time t+1
    dvdt = -lam * v - O_f @ s + O_s @ r + (O_c @ r + F_c @ D[i:] @ r) - (O_k @ r + F_k @ C @ y)
    dvdt = dvdt + (D[i:].T @ ((lam*x_des)+Dx)) - (D[i:].T @ D[i:] @ s)
    v_next = v + dvdt*dt + np.sqrt(dt)*sigma*np.random.randn(len(dvdt))

    # check if there are neurons whose voltage is above threshold
    above = np.where(v_next > T)[0]

    # introduce a control to let only one neuron fire at the time
    s_next=np.zeros(s.shape)
    if len(above):
        s_next[np.argmax(v_next)] = 1/dt

    # update rate
    drdt = s_next - lam*r
    r_next = r + drdt*dt
    
    return r_next, s_next, v_next, u_next

"""
Function for running a single step of the idealized Kalman filter.
"""
def run_Kfidealized_step(x_hat,A,B,u,Kf,y,C,dt):

    dxdt = A@x_hat + B@u + Kf@(y-(C@x_hat))
    x_next = x_hat + dxdt*dt
    
    return x_next

"""
Function for running a single step of a linearized Dynamical System (DS).
"""
def run_DSlinearized_step(x,A,B,u,dist,dt):
    
    dxdt = A@x + B@u 
    x_next = x + dxdt*dt + np.sqrt(dt)*dist
    
    return x_next

"""
Function for running a single step of a (non-linear) simulated Cartpole Dynamical System.
"""
def run_Cartpolereal_step(x,u,dist,m,M,L,g,d,dt):
    Sy=np.sin(x[2])
    Cy=np.cos(x[2])

    D = m*L*L*(M+m*(1-Cy**2))
    
    dy_1 = x[1]
    dy_2 = (1/D)*(-m**2*L**2*g*Cy*Sy + m*L**2*(m*L*x[3]**2*Sy - d*x[1])) + m*L*L*(1/D)*u
    dy_3 = x[3]
    dy_4 = (1/D)*((m+M)*m*g*L*Sy - m*L*Cy*(m*L*x[3]**2*Sy - d*x[1])) - m*L*Cy*(1/D)*u
    
    dxdt=np.array([dy_1, dy_2, dy_3, dy_4])

    x_next=x+dxdt*dt + np.sqrt(dt)*dist
    
    return x_next