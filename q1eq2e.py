
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from scipy.integrate import odeint
from scipy.optimize import fsolve
from pylab import *


# In[3]:

##multi-step reference signal

def heaviside(the_t):
    """ step function """
    if (the_t <= 300):
        the_ref = 0.5
    elif (300 < the_t <= 700):
        the_ref = 1.5
    elif (700 < the_t < 1100):
        the_ref = 5.0
    elif (1100 <= the_t <= 1500):
        the_ref = 1.0
    else:
        the_ref = 0
    return the_ref


xval = np.linspace(0,1500,100)
zval = []
for val in xval:
    b = heaviside(val)
    zval.append(b)
    
plt.figure()
plt.plot(xval, zval)
plt.xlabel('time')
plt.ylabel('[ref]')
plt.show()


# In[5]:

##just normal ode
##variables
X0 = 0.1
Xx = 2
Yx = 1.4
nx = 2
Kr = 1
Kq = 0.1
Kb = 0.5
Kc = 0.015
Kd = 0.5
KQ = 0.05
diff_cell = 2
gamma_i = 0.4
gamma_e = 0.2
diff_ex = 13.3
##800 micrometer^2/sec



def eq1 (Y, t, Ref):
    A_Q2 = Y[0]
    B = Y[1]
    C = Y[2]
    D = Y[3]
    Q1c = Y[4]
   ## Q1e = Y[5]
    Q1t = Y[5]
    Q2c = Y[6]
  ##  Q2e = Y[8]
    Q2t = Y[7]
    dydt =np.empty(len(Y))
    
    ##A_Q2
    dydt[0] = (X0 + (Xx * ( (Kr ** nx) / ((Kr ** nx) + (Ref ** nx)) ))) * (X0 + (Xx * ( (Q2c ** nx) / ((Kq ** nx) + (Q2c ** nx)) ))) - (Yx * A_Q2)  
    ##B
    dydt[1] = X0 + (Xx * ((A_Q2 ** nx)/((Kb ** nx) +(A_Q2 ** nx)))) - (Yx * B)
    ##C
    dydt[2] = X0 + (Xx * ((Q1t ** nx)/((Kc ** nx) +(Q1t ** nx)))) - (Yx * C)
    ##D
    dydt[3] = X0 + (Xx * ((Kd ** nx) / ((Kd ** nx) + (C ** nx)))) - (Yx * D)
    ##Q1c
    dydt[4] = KQ * B + (diff_cell * (Q1c - Q1c)) - (gamma_i*Q1c)
    ##Q1t
    dydt[5] = (diff_cell * (Q1c - Q1t)) - (gamma_i*Q1t)
    ##Q1e
    ##dydt[6] = (diff_cell * (Q1c - Q1c)) + (diff_cell * (Q1t - Q1c)) - (gamma_e*Q1c) + (diff_ex * Q1c)
    ##Q2c
    dydt[6] = (diff_cell * (Q2t - Q2c)) - (gamma_i*Q2c)
    ##Q2t
    dydt[7] = (KQ*D) +(diff_cell * (Q2t - Q2t)) - (gamma_i*Q2t)
    ##Q2e
    ##dydt[9] = (diff_cell * (Q2c - Q2t)) + (diff_cell * (Q2t - Q2t)) - (gamma_e*Q2c)+ (diff_ex * Q2t)

    return dydt

Y_initial= np.array([0,0,0,0,0,0,0,0])
Y_initial_nonZero= np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
t = np.linspace(0,400,100000)
r100 = 100 ##[Ref]
r05 = 0.5
r0 = 0
##eq1 = odeint(eq1, Y_initial, t, args=(r,), mxstep=5000000)
eqr100 = odeint(eq1, Y_initial_nonZero, t, args=(r100,), mxstep=5000000)
eqr05 = odeint(eq1, Y_initial_nonZero, t, args=(r05,), mxstep=5000000)
eqr0 = odeint(eq1, Y_initial_nonZero, t, args=(r0,), mxstep=5000000)

e0qr100 = odeint(eq1, Y_initial, t, args=(r100,), mxstep=5000000)
e0qr05 = odeint(eq1, Y_initial, t, args=(r05,), mxstep=5000000)
e0qr0 = odeint(eq1, Y_initial, t, args=(r0,), mxstep=5000000)
##print(e0qr0)

plt.figure()
plt.plot(t, e0qr100[:,3], label = '[D], ref=100')
plt.plot(t, e0qr05[:,3], label = '[D], ref=0.5')
plt.plot(t, e0qr0[:,3], label = '[D], ref=0')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('conc.')
plt.title('initial conc = 0')
plt.show()


plt.figure()
##plt.plot(t, eqr100[:,3], label = '[D], ref=100')
##plt.plot(t, eqr100[:,2], label = '[C], ref=100')
plt.plot(t, eqr100[:,0], label = '[A_Q2], ref=100')
##plt.plot(t, eqr05[:,3], label = '[D], ref=0.5')
##plt.plot(t, eqr05[:,2], label = '[C], ref=0.5')
plt.plot(t, eqr05[:,0], label = '[A_Q2], ref=0.5')
##plt.plot(t, eqr0[:,3], label = '[D], ref=0')
##plt.plot(t, eqr0[:,2], label = '[C], ref=0')
plt.plot(t, eqr0[:,0], label = '[A_Q2], ref=0')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('conc.')
plt.title('initial conc = 0.5')
plt.show()


# In[13]:

##normal ode + reference changes as multi-step

t1 = np.linspace(0,300,100)
t2 = np.linspace(300,700,100)
t3 = np.linspace(700,1100,100)
t4 = np.linspace(1100,1500,100)

eq1_noQe = odeint(eq1, Y_initial, t1, args=(0.5,), mxstep=5000000)
results1 = eq1_noQe[-1:,]
b_results1 = results1.ravel()
#print(results1)
#print(b_results1)


eq2_noQe = odeint(eq1, b_results1, t2, args=(1.5,), mxstep=5000000)
results2 = eq2_noQe[-1:,]
b_results2 = results2.ravel()
##print(results2)
##print(b_results2)

eq3_noQe = odeint(eq1, b_results2, t3, args=(5,), mxstep=5000000)
results3 = eq3_noQe[-1:,]
b_results3 = results3.ravel()
##print(results3)
##print(b_results3)

eq4_noQe = odeint(eq1, b_results3, t4, args=(1,), mxstep=5000000)
results4 = eq4_noQe[-1:,]
b_results4 = results4.ravel()
##print(results4)
##print(b_results4)


plt.figure()
plt.plot(t1, eq1_noQe[:,3], label = 'ref=0.5')
plt.plot(t2, eq2_noQe[:,3], label = 'ref=1.5')
plt.plot(t3, eq3_noQe[:,3], label = 'ref=5')
plt.plot(t4, eq4_noQe[:,3], label = 'ref=1')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('conc.')
plt.title('initial conc = 0')
plt.show()



# In[14]:

##normal ode + reference changes as multi-step
##Y_initial_nonZero= np.array([0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25])
t1 = np.linspace(0,300,100)
t2 = np.linspace(300,700, 100)
t3 = np.linspace(700,1100, 100)
t4 = np.linspace(1100,1500, 100)

eq10_noQe = odeint(eq1, Y_initial_nonZero, t1, args=(0.5,), mxstep=5000000)
results10 = eq10_noQe[-1:,]
b_results10 = results10.ravel()
#print(results10)
#print(b_results10)


eq20_noQe = odeint(eq1, b_results10, t2, args=(1.5,), mxstep=5000000)
results20 = eq20_noQe[-1:,]
b_results20 = results20.ravel()
##print(results20)
##print(b_results20)

eq30_noQe = odeint(eq1, b_results20, t3, args=(5,), mxstep=5000000)
results30 = eq30_noQe[-1:,]
b_results30 = results30.ravel()
##print(results30)
##print(b_results30)

eq40_noQe = odeint(eq1, b_results30, t4, args=(1,), mxstep=5000000)
results40 = eq4_noQe[-1:,]
b_results40 = results40.ravel()
##print(results40)
##print(b_results40)


plt.figure()


plt.plot(t1, eq10_noQe[:,3], label = 'ref=0.5, not zero')
plt.plot(t2, eq20_noQe[:,3], label = 'ref=1.5, not zero')
plt.plot(t3, eq30_noQe[:,3], label = 'ref=5, not zero')
plt.plot(t4, eq40_noQe[:,3], label = 'ref=1, not zero')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('conc.')
plt.title('initial conc = 0.5')




plt.show()


# In[74]:

plt.figure()



plt.plot(t1, eq1_noQe[:,3], 'b', label = 'ref=0.5')
plt.plot(t2, eq2_noQe[:,3], 'b', label = 'ref=1.5')
plt.plot(t3, eq3_noQe[:,3], 'b', label = 'ref=5')
plt.plot(t4, eq4_noQe[:,3], 'b', label = 'ref=1')
##plt.legend(loc='best')

plt.plot(t1, eq10_noQe[:,3], 'r',  alpha=0.7, label = 'ref=0.5, not zero')
plt.plot(t2, eq20_noQe[:,3], 'r',  alpha=0.7, label = 'ref=1.5, not zero')
plt.plot(t3, eq30_noQe[:,3], 'r',  alpha=0.7, label = 'ref=5, not zero')
plt.plot(t4, eq40_noQe[:,3], 'r',  alpha=0.7, label = 'ref=1, not zero')
##plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('[D]')
##plt.title('initial conc = 0.5')


plt.show()


# In[182]:

##just vode ode
##http://modelling3e4.connectmv.com/wiki/Software_tutorial/Integration_of_ODEs
import numpy as np
from scipy import integrate
from matplotlib.pylab import *


def tank(t, y):   
    Ref = 0.5
    X0 = 0.1
    Xx = 2
    Yx = 1.4
    nx = 2
    Kr = 1
    Kq = 0.1
    Kb = 0.5
    Kc = 0.015
    Kd = 0.5
    KQ = 0.05
    diff_cell = 2
    gamma_i = 0.4
    gamma_e = 0.2
    diff_ex = 13.3 ##800 micrometer^2/sec
    
    #Assign some variables for convenience of notation
    A_Q2 = y[0]
    B = y[1]
    C = y[2]
    D = y[3]
    Q1c = y[4]
    ##Q1e = y[5]
    Q1t = y[5]
    Q2c = y[6]
    ##Q2e = y[8]
    Q2t = y[7]
      
    ## Output from ODE function must be a COLUMN vector, with n rows
    n = len(y) # 9: implies we have ten ODEs
    
    dydt = np.zeros((n,1))
    
    ##A_Q2
    dydt[0] = (X0 + (Xx * ( (Kr ** nx) / ((Kr ** nx) + (Ref ** nx)) ))) * (X0 + (Xx * ( (Q2c ** nx) / ((Kq ** nx) + (Q2c ** nx)) ))) - (Yx * A_Q2)  
    ##B
    dydt[1] = X0 + (Xx * ((A_Q2 ** nx)/((Kb ** nx) +(A_Q2 ** nx)))) - (Yx * B)
    ##C
    dydt[2] = X0 + (Xx * ((Q1t ** nx)/((Kc ** nx) +(Q1t ** nx)))) - (Yx * C)
    ##D
    dydt[3] = X0 + (Xx * ((Kd ** nx) / ((Kd ** nx) + (C ** nx)))) - (Yx * D)
    ##Q1c
    dydt[4] = KQ * B + (diff_cell * (Q1c - Q1c)) - (gamma_i*Q1c)
    ##Q1t
    dydt[5] = (diff_cell * (Q1c - Q1t)) - (gamma_i*Q1t)
    ##Q1e
    ##dydt[6] = (diff_cell * (Q1c - Q1c)) + (diff_cell * (Q1t - Q1c)) - (gamma_e*Q1c) + (diff_ex * Q1c)
    ##Q2c
    dydt[6] = (diff_cell * (Q2t - Q2c)) - (gamma_i*Q2c)
    ##Q2t
    dydt[7] = (KQ*D) +(diff_cell * (Q2t - Q2t)) - (gamma_i*Q2t)
    ##Q2e
    ##dydt[9] = (diff_cell * (Q2c - Q2t)) + (diff_cell * (Q2t - Q2t)) - (gamma_e*Q2c)+ (diff_ex * Q2t)
    
   
    return dydt

# The ``driver`` that will integrate the ODE(s):
if __name__ == '__main__':
    
    # Start by specifying the integrator:
    # use ``vode`` with "backward differentiation formula"
    r = integrate.ode(tank).set_integrator('vode', method='bdf')
    
    # Set the time range
    t_start = 0.0
    t_final = 400.0
    delta_t = 0.05
    
    # Number of time steps: 1 extra for initial condition
    num_steps = int(np.floor((t_final - t_start)/delta_t) + 1)
    
    # Set initial condition(s): for integrating variable and time!
    A_Q2_t_zero = 0.5
    B_t_zero = 0.5
    C_t_zero = 0.5
    D_t_zero = 0.5
    Q1c_t_zero = 0.5
    Q1e_t_zero = 0.5
    Q1t_t_zero = 0.5
    Q2c_t_zero = 0.5
    Q2e_t_zero = 0.5
    Q2t_t_zero = 0.5
    r.set_initial_value([A_Q2_t_zero, B_t_zero, C_t_zero, D_t_zero, Q1c_t_zero, Q1e_t_zero, Q1t_t_zero, Q2c_t_zero, Q2e_t_zero, Q2t_t_zero], t_start)
    
    # Additional Python step: create vectors to store trajectories
    t = np.zeros((num_steps, 1))
    D = np.zeros((num_steps, 1))
    
    A_Q2 = np.zeros((num_steps, 1))
    B = np.zeros((num_steps, 1))
    C = np.zeros((num_steps, 1))
    Q1c = np.zeros((num_steps, 1))
    Q1t = np.zeros((num_steps, 1))
    Q1e = np.zeros((num_steps, 1))
    Q2c = np.zeros((num_steps, 1))
    Q2t = np.zeros((num_steps, 1))
    Q2e = np.zeros((num_steps, 1))
    
    t[0] = t_start
    D[0] = D_t_zero
    
    A_Q2[0] = A_Q2_t_zero
    B[0] = B_t_zero
    C[0] = C_t_zero
    Q1c[0] = Q1c_t_zero
    Q1e [0] = Q1e_t_zero
    Q1t[0] = Q1t_t_zero
    Q2c[0] = Q2c_t_zero 
    Q2e[0] = Q2e_t_zero 
    Q2t[0] = Q2t_t_zero 

    
    # Integrate the ODE(s) across each delta_t timestep
    k = 1
    while r.successful() and k < num_steps:
        r.integrate(r.t + delta_t)
        # Store the results to plot later
        t[k] = r.t
        D[k] = r.y[3]
        k += 1
        
    # All done!  Plot the trajectories in two separate plots:
    fig = figure()
    ax1 = subplot(211)
    ax1.plot(t, D)
    ax1.set_xlim(t_start, t_final)
    ax1.set_xlabel('Time [minutes]')
    ax1.set_ylabel('Concentration [mol/L]')
    ax1.set_title("initial conc = 0.5")
    ax1.grid('on')
    fig.savefig('coupled-ode-Python.png')
    
d1_ceu = D
d1 = D[-1]
aq21 = A_Q2[-1]
c1 = C[-1]
b1 = B[-1]
q1c1 = Q1c[-1]
q1t1 = Q1t[-1]
q2c1 = Q2c[-1]
q2t1 = Q2t[-1] 


# In[172]:

##just vode ode
##http://modelling3e4.connectmv.com/wiki/Software_tutorial/Integration_of_ODEs
import numpy as np
from scipy import integrate
from matplotlib.pylab import *


def tank1(t, y):   
    Ref = 0.5
    X0 = 0.1
    Xx = 2
    Yx = 1.4
    nx = 2
    Kr = 1
    Kq = 0.1
    Kb = 0.5
    Kc = 0.015
    Kd = 0.5
    KQ = 0.05
    diff_cell = 2
    gamma_i = 0.4
    gamma_e = 0.2
    diff_ex = 13.3 ##800 micrometer^2/sec
    
    #Assign some variables for convenience of notation
    A_Q2 = y[0]
    B = y[1]
    C = y[2]
    D = y[3]
    Q1c = y[4]
    ##Q1e = y[5]
    Q1t = y[5]
    Q2c = y[6]
    ##Q2e = y[8]
    Q2t = y[7]
      
    ## Output from ODE function must be a COLUMN vector, with n rows
    n = len(y) # 9: implies we have ten ODEs
    
    dydt = np.zeros((n,1))
    
    ##A_Q2
    dydt[0] = (X0 + (Xx * ( (Kr ** nx) / ((Kr ** nx) + (Ref ** nx)) ))) * (X0 + (Xx * ( (Q2c ** nx) / ((Kq ** nx) + (Q2c ** nx)) ))) - (Yx * A_Q2)  
    ##B
    dydt[1] = X0 + (Xx * ((A_Q2 ** nx)/((Kb ** nx) +(A_Q2 ** nx)))) - (Yx * B)
    ##C
    dydt[2] = X0 + (Xx * ((Q1t ** nx)/((Kc ** nx) +(Q1t ** nx)))) - (Yx * C)
    ##D
    dydt[3] = X0 + (Xx * ((Kd ** nx) / ((Kd ** nx) + (C ** nx)))) - (Yx * D)
    ##Q1c
    dydt[4] = KQ * B + (diff_cell * (Q1c - Q1c)) - (gamma_i*Q1c)
    ##Q1t
    dydt[5] = (diff_cell * (Q1c - Q1t)) - (gamma_i*Q1t)
    ##Q1e
    ##dydt[6] = (diff_cell * (Q1c - Q1c)) + (diff_cell * (Q1t - Q1c)) - (gamma_e*Q1c) + (diff_ex * Q1c)
    ##Q2c
    dydt[6] = (diff_cell * (Q2t - Q2c)) - (gamma_i*Q2c)
    ##Q2t
    dydt[7] = (KQ*D) +(diff_cell * (Q2t - Q2t)) - (gamma_i*Q2t)
    ##Q2e
    ##dydt[9] = (diff_cell * (Q2c - Q2t)) + (diff_cell * (Q2t - Q2t)) - (gamma_e*Q2c)+ (diff_ex * Q2t)
    
   
    return dydt

# The ``driver`` that will integrate the ODE(s):
if __name__ == '__main__':
    
    # Start by specifying the integrator:
    # use ``vode`` with "backward differentiation formula"
    r = integrate.ode(tank1).set_integrator('vode', method='bdf')
    
    # Set the time range
    t_start = 0.0
    t_final = 300.0
    delta_t = 0.05
    
    # Number of time steps: 1 extra for initial condition
    num_steps = int(np.floor((t_final - t_start)/delta_t) + 1)
    
    # Set initial condition(s): for integrating variable and time!
    A_Q2_t_zero = 0
    B_t_zero = 0
    C_t_zero = 0
    D_t_zero = 0
    Q1c_t_zero = 0
    
    Q1t_t_zero = 0
    Q2c_t_zero = 0
    
    Q2t_t_zero = 0
    r.set_initial_value([A_Q2_t_zero, B_t_zero, C_t_zero, D_t_zero, Q1c_t_zero, Q1t_t_zero, Q2c_t_zero, Q2t_t_zero], t_start)
    
    # Additional Python step: create vectors to store trajectories
    t = np.zeros((num_steps, 1))
    D = np.zeros((num_steps, 1))
    
    A_Q2 = np.zeros((num_steps, 1))
    B = np.zeros((num_steps, 1))
    C = np.zeros((num_steps, 1))
    Q1c = np.zeros((num_steps, 1))
    Q1t = np.zeros((num_steps, 1))
    Q2c = np.zeros((num_steps, 1))
    Q2t = np.zeros((num_steps, 1))
    
    
    t[0] = t_start
    D[0] = D_t_zero
    
    A_Q2[0] = A_Q2_t_zero
    B[0] = B_t_zero
    C[0] = C_t_zero
    Q1c[0] = Q1c_t_zero
    Q1t[0] = Q1t_t_zero
    Q2c[0] = Q2c_t_zero 
    Q2t[0] = Q2t_t_zero 

    
    # Integrate the ODE(s) across each delta_t timestep
    k = 1
    while r.successful() and k < num_steps:
        r.integrate(r.t + delta_t)
        # Store the results to plot later
        t[k] = r.t
        D[k] = r.y[3]
        k += 1
        
    # All done!  Plot the trajectories in two separate plots:
    fig = figure()
    ax1 = subplot(211)
    ax1.plot(t, D)
    ax1.set_xlim(t_start, t_final)
    ax1.set_xlabel('Time [minutes]')
    ax1.set_ylabel('Concentration [mol/L]')
    ax1.grid('on')
    ax1.set_title("initial conc = 0.0")
    fig.savefig('coupled-ode-Python.png')
d1_ceu = D
d1 = D[-1]
aq21 = A_Q2[-1]
c1 = C[-1]
b1 = B[-1]
q1c1 = Q1c[-1]
q1t1 = Q1t[-1]
q2c1 = Q2c[-1]
q2t1 = Q2t[-1] 



# In[183]:

##vode ode + reference changes as multi-step

def tank2(t, y):   
    Ref = 1.5
    X0 = 0.1
    Xx = 2
    Yx = 1.4
    nx = 2
    Kr = 1
    Kq = 0.1
    Kb = 0.5
    Kc = 0.015
    Kd = 0.5
    KQ = 0.05
    diff_cell = 2
    gamma_i = 0.4
    gamma_e = 0.2
    diff_ex = 13.3 ##800 micrometer^2/sec
    
    #Assign some variables for convenience of notation
    A_Q2 = y[0]
    B = y[1]
    C = y[2]
    D = y[3]
    Q1c = y[4]
    ##Q1e = y[5]
    Q1t = y[5]
    Q2c = y[6]
    ##Q2e = y[8]
    Q2t = y[7]
      
    ## Output from ODE function must be a COLUMN vector, with n rows
    n = len(y) # 9: implies we have ten ODEs
    
    dydt = np.zeros((n,1))
    
    ##A_Q2
    dydt[0] = (X0 + (Xx * ( (Kr ** nx) / ((Kr ** nx) + (Ref ** nx)) ))) * (X0 + (Xx * ( (Q2c ** nx) / ((Kq ** nx) + (Q2c ** nx)) ))) - (Yx * A_Q2)  
    ##B
    dydt[1] = X0 + (Xx * ((A_Q2 ** nx)/((Kb ** nx) +(A_Q2 ** nx)))) - (Yx * B)
    ##C
    dydt[2] = X0 + (Xx * ((Q1t ** nx)/((Kc ** nx) +(Q1t ** nx)))) - (Yx * C)
    ##D
    dydt[3] = X0 + (Xx * ((Kd ** nx) / ((Kd ** nx) + (C ** nx)))) - (Yx * D)
    ##Q1c
    dydt[4] = KQ * B + (diff_cell * (Q1c - Q1c)) - (gamma_i*Q1c)
    ##Q1t
    dydt[5] = (diff_cell * (Q1c - Q1t)) - (gamma_i*Q1t)
    ##Q1e
    ##dydt[6] = (diff_cell * (Q1c - Q1c)) + (diff_cell * (Q1t - Q1c)) - (gamma_e*Q1c) + (diff_ex * Q1c)
    ##Q2c
    dydt[6] = (diff_cell * (Q2t - Q2c)) - (gamma_i*Q2c)
    ##Q2t
    dydt[7] = (KQ*D) +(diff_cell * (Q2t - Q2t)) - (gamma_i*Q2t)
    ##Q2e
    ##dydt[9] = (diff_cell * (Q2c - Q2t)) + (diff_cell * (Q2t - Q2t)) - (gamma_e*Q2c)+ (diff_ex * Q2t)
    
   
    return dydt

# The ``driver`` that will integrate the ODE(s):
if __name__ == '__main__':
    
    # Start by specifying the integrator:
    # use ``vode`` with "backward differentiation formula"
    r = integrate.ode(tank2).set_integrator('vode', method='bdf')
    
    # Set the time range
    t_start2 = 300.0
    t_final2 = 700.0
    delta_t = 0.05
    
    # Number of time steps: 1 extra for initial condition
    num_steps = int(np.floor((t_final - t_start)/delta_t) + 1)
    
    # Set initial condition(s): for integrating variable and time!
    A_Q2_t_zero = aq21
    B_t_zero = b1
    C_t_zero = c1
    D_t_zero = d1
    Q1c_t_zero = q1c1
   
    Q1t_t_zero = q1t1
    Q2c_t_zero = q2c1
    
    Q2t_t_zero = q2t1
    r.set_initial_value([A_Q2_t_zero, B_t_zero, C_t_zero, D_t_zero, Q1c_t_zero, Q1t_t_zero, Q2c_t_zero, Q2t_t_zero], t_start)
    
    # Additional Python step: create vectors to store trajectories
    t = np.zeros((num_steps, 1))
    D = np.zeros((num_steps, 1))
    
    A_Q2 = np.zeros((num_steps, 1))
    B = np.zeros((num_steps, 1))
    C = np.zeros((num_steps, 1))
    Q1c = np.zeros((num_steps, 1))
    Q1t = np.zeros((num_steps, 1))
    Q1e = np.zeros((num_steps, 1))
    Q2c = np.zeros((num_steps, 1))
    Q2t = np.zeros((num_steps, 1))
    Q2e = np.zeros((num_steps, 1))
    
    t[0] = t_start2
    D[0] = D_t_zero
    
    A_Q2[0] = A_Q2_t_zero
    B[0] = B_t_zero
    C[0] = C_t_zero
    Q1c[0] = Q1c_t_zero
    Q1e [0] = Q1e_t_zero
    Q1t[0] = Q1t_t_zero
    Q2c[0] = Q2c_t_zero 
    Q2e[0] = Q2e_t_zero 
    Q2t[0] = Q2t_t_zero 

    
    # Integrate the ODE(s) across each delta_t timestep
    k = 1
    while r.successful() and k < num_steps:
        r.integrate(r.t + delta_t)
        # Store the results to plot later
        t[k] = r.t
        D[k] = r.y[3]
        k += 1
        
    # All done!  Plot the trajectories in two separate plots:
    fig = figure()
    ax1 = subplot(211)
    ax1.plot(t, D)
    ax1.set_xlim(t_start2, t_final2)
    ax1.set_xlabel('Time [minutes]')
    ax1.set_ylabel('Concentration [mol/L]')
    ax1.grid('on')
    ##ax1.set_title("initial conc = 0.0")
    fig.savefig('coupled-ode-Python.png')

d2_ceu = D
d2 = D[-1]
aq22 = A_Q2[-1]
c2 = C[-1]
b2 = B[-1]
q1c2 = Q1c[-1]
q1t2 = Q1t[-1]
q2c2 = Q2c[-1]
q2t2 = Q2t[-1] 


# In[184]:

##vode ode + reference changes as multi-step

def tank3(t, y):   
    Ref = 5
    X0 = 0.1
    Xx = 2
    Yx = 1.4
    nx = 2
    Kr = 1
    Kq = 0.1
    Kb = 0.5
    Kc = 0.015
    Kd = 0.5
    KQ = 0.05
    diff_cell = 2
    gamma_i = 0.4
    gamma_e = 0.2
    diff_ex = 13.3 ##800 micrometer^2/sec
    
    #Assign some variables for convenience of notation
    A_Q2 = y[0]
    B = y[1]
    C = y[2]
    D = y[3]
    Q1c = y[4]
    ##Q1e = y[5]
    Q1t = y[5]
    Q2c = y[6]
    ##Q2e = y[8]
    Q2t = y[7]
      
    ## Output from ODE function must be a COLUMN vector, with n rows
    n = len(y) # 9: implies we have ten ODEs
    
    dydt = np.zeros((n,1))
    
    ##A_Q2
    dydt[0] = (X0 + (Xx * ( (Kr ** nx) / ((Kr ** nx) + (Ref ** nx)) ))) * (X0 + (Xx * ( (Q2c ** nx) / ((Kq ** nx) + (Q2c ** nx)) ))) - (Yx * A_Q2)  
    ##B
    dydt[1] = X0 + (Xx * ((A_Q2 ** nx)/((Kb ** nx) +(A_Q2 ** nx)))) - (Yx * B)
    ##C
    dydt[2] = X0 + (Xx * ((Q1t ** nx)/((Kc ** nx) +(Q1t ** nx)))) - (Yx * C)
    ##D
    dydt[3] = X0 + (Xx * ((Kd ** nx) / ((Kd ** nx) + (C ** nx)))) - (Yx * D)
    ##Q1c
    dydt[4] = KQ * B + (diff_cell * (Q1c - Q1c)) - (gamma_i*Q1c)
    ##Q1t
    dydt[5] = (diff_cell * (Q1c - Q1t)) - (gamma_i*Q1t)
    ##Q1e
    ##dydt[6] = (diff_cell * (Q1c - Q1c)) + (diff_cell * (Q1t - Q1c)) - (gamma_e*Q1c) + (diff_ex * Q1c)
    ##Q2c
    dydt[6] = (diff_cell * (Q2t - Q2c)) - (gamma_i*Q2c)
    ##Q2t
    dydt[7] = (KQ*D) +(diff_cell * (Q2t - Q2t)) - (gamma_i*Q2t)
    ##Q2e
    ##dydt[9] = (diff_cell * (Q2c - Q2t)) + (diff_cell * (Q2t - Q2t)) - (gamma_e*Q2c)+ (diff_ex * Q2t)
    
   
    return dydt

# The ``driver`` that will integrate the ODE(s):
if __name__ == '__main__':
    
    # Start by specifying the integrator:
    # use ``vode`` with "backward differentiation formula"
    r = integrate.ode(tank3).set_integrator('vode', method='bdf')
    
    # Set the time range
    t_start2 = 700.0
    t_final2 = 1100.0
    delta_t = 0.05
    
    # Number of time steps: 1 extra for initial condition
    num_steps = int(np.floor((t_final - t_start)/delta_t) + 1)
    
    # Set initial condition(s): for integrating variable and time!
    A_Q2_t_zero = aq22
    B_t_zero = b2
    C_t_zero = c2
    D_t_zero = d2
    Q1c_t_zero = q1c2
   
    Q1t_t_zero = q1t2
    Q2c_t_zero = q2c2
    
    Q2t_t_zero = q2t2
    r.set_initial_value([A_Q2_t_zero, B_t_zero, C_t_zero, D_t_zero, Q1c_t_zero, Q1t_t_zero, Q2c_t_zero, Q2t_t_zero], t_start)
    
    # Additional Python step: create vectors to store trajectories
    t = np.zeros((num_steps, 1))
    D = np.zeros((num_steps, 1))
    
    A_Q2 = np.zeros((num_steps, 1))
    B = np.zeros((num_steps, 1))
    C = np.zeros((num_steps, 1))
    Q1c = np.zeros((num_steps, 1))
    Q1t = np.zeros((num_steps, 1))
    Q1e = np.zeros((num_steps, 1))
    Q2c = np.zeros((num_steps, 1))
    Q2t = np.zeros((num_steps, 1))
    Q2e = np.zeros((num_steps, 1))
    
    t[0] = t_start2
    D[0] = D_t_zero
    
    A_Q2[0] = A_Q2_t_zero
    B[0] = B_t_zero
    C[0] = C_t_zero
    Q1c[0] = Q1c_t_zero
    Q1e [0] = Q1e_t_zero
    Q1t[0] = Q1t_t_zero
    Q2c[0] = Q2c_t_zero 
    Q2e[0] = Q2e_t_zero 
    Q2t[0] = Q2t_t_zero 

    
    # Integrate the ODE(s) across each delta_t timestep
    k = 1
    while r.successful() and k < num_steps:
        r.integrate(r.t + delta_t)
        # Store the results to plot later
        t[k] = r.t
        D[k] = r.y[3]
        k += 1
        
    # All done!  Plot the trajectories in two separate plots:
    fig = figure()
    ax1 = subplot(211)
    ax1.plot(t, D)
    ax1.set_xlim(t_start2, t_final2)
    ax1.set_xlabel('Time [minutes]')
    ax1.set_ylabel('Concentration [mol/L]')
    ax1.grid('on')
    ##ax1.set_title("initial conc = 0.0")
    fig.savefig('coupled-ode-Python.png')

d3_ceu = D
d3 = D[-1]
aq23 = A_Q2[-1]
c3 = C[-1]
b3 = B[-1]
q1c3 = Q1c[-1]
q1t3 = Q1t[-1]
q2c3 = Q2c[-1]
q2t3 = Q2t[-1] 


# In[186]:

##vode ode + reference changes as multi-step

def tank4(t, y):   
    Ref = 1.0
    X0 = 0.1
    Xx = 2
    Yx = 1.4
    nx = 2
    Kr = 1
    Kq = 0.1
    Kb = 0.5
    Kc = 0.015
    Kd = 0.5
    KQ = 0.05
    diff_cell = 2
    gamma_i = 0.4
    gamma_e = 0.2
    diff_ex = 13.3 ##800 micrometer^2/sec
    
    #Assign some variables for convenience of notation
    A_Q2 = y[0]
    B = y[1]
    C = y[2]
    D = y[3]
    Q1c = y[4]
    ##Q1e = y[5]
    Q1t = y[5]
    Q2c = y[6]
    ##Q2e = y[8]
    Q2t = y[7]
      
    ## Output from ODE function must be a COLUMN vector, with n rows
    n = len(y) # 9: implies we have ten ODEs
    
    dydt = np.zeros((n,1))
    
    ##A_Q2
    dydt[0] = (X0 + (Xx * ( (Kr ** nx) / ((Kr ** nx) + (Ref ** nx)) ))) * (X0 + (Xx * ( (Q2c ** nx) / ((Kq ** nx) + (Q2c ** nx)) ))) - (Yx * A_Q2)  
    ##B
    dydt[1] = X0 + (Xx * ((A_Q2 ** nx)/((Kb ** nx) +(A_Q2 ** nx)))) - (Yx * B)
    ##C
    dydt[2] = X0 + (Xx * ((Q1t ** nx)/((Kc ** nx) +(Q1t ** nx)))) - (Yx * C)
    ##D
    dydt[3] = X0 + (Xx * ((Kd ** nx) / ((Kd ** nx) + (C ** nx)))) - (Yx * D)
    ##Q1c
    dydt[4] = KQ * B + (diff_cell * (Q1c - Q1c)) - (gamma_i*Q1c)
    ##Q1t
    dydt[5] = (diff_cell * (Q1c - Q1t)) - (gamma_i*Q1t)
    ##Q1e
    ##dydt[6] = (diff_cell * (Q1c - Q1c)) + (diff_cell * (Q1t - Q1c)) - (gamma_e*Q1c) + (diff_ex * Q1c)
    ##Q2c
    dydt[6] = (diff_cell * (Q2t - Q2c)) - (gamma_i*Q2c)
    ##Q2t
    dydt[7] = (KQ*D) +(diff_cell * (Q2t - Q2t)) - (gamma_i*Q2t)
    ##Q2e
    ##dydt[9] = (diff_cell * (Q2c - Q2t)) + (diff_cell * (Q2t - Q2t)) - (gamma_e*Q2c)+ (diff_ex * Q2t)
    
   
    return dydt

# The ``driver`` that will integrate the ODE(s):
if __name__ == '__main__':
    
    # Start by specifying the integrator:
    # use ``vode`` with "backward differentiation formula"
    r = integrate.ode(tank4).set_integrator('vode', method='bdf')
    
    # Set the time range
    t_start2 = 1100.0
    t_final2 = 1500.0
    delta_t = 0.05
    
    # Number of time steps: 1 extra for initial condition
    num_steps = int(np.floor((t_final - t_start)/delta_t) + 1)
    
    # Set initial condition(s): for integrating variable and time!
    A_Q2_t_zero = aq23
    B_t_zero = b3
    C_t_zero = c3
    D_t_zero = d3
    Q1c_t_zero = q1c3
   
    Q1t_t_zero = q1t3
    Q2c_t_zero = q2c3
    
    Q2t_t_zero = q2t3
    r.set_initial_value([A_Q2_t_zero, B_t_zero, C_t_zero, D_t_zero, Q1c_t_zero, Q1t_t_zero, Q2c_t_zero, Q2t_t_zero], t_start)
    
    # Additional Python step: create vectors to store trajectories
    t = np.zeros((num_steps, 1))
    D = np.zeros((num_steps, 1))
    
    A_Q2 = np.zeros((num_steps, 1))
    B = np.zeros((num_steps, 1))
    C = np.zeros((num_steps, 1))
    Q1c = np.zeros((num_steps, 1))
    Q1t = np.zeros((num_steps, 1))
    Q1e = np.zeros((num_steps, 1))
    Q2c = np.zeros((num_steps, 1))
    Q2t = np.zeros((num_steps, 1))
    Q2e = np.zeros((num_steps, 1))
    
    t[0] = t_start2
    D[0] = D_t_zero
    
    A_Q2[0] = A_Q2_t_zero
    B[0] = B_t_zero
    C[0] = C_t_zero
    Q1c[0] = Q1c_t_zero
    Q1e [0] = Q1e_t_zero
    Q1t[0] = Q1t_t_zero
    Q2c[0] = Q2c_t_zero 
    Q2e[0] = Q2e_t_zero 
    Q2t[0] = Q2t_t_zero 

    
    # Integrate the ODE(s) across each delta_t timestep
    k = 1
    while r.successful() and k < num_steps:
        r.integrate(r.t + delta_t)
        # Store the results to plot later
        t[k] = r.t
        D[k] = r.y[3]
        k += 1
    

d4_ceu = D 


# In[190]:

t1 =  np.linspace(0, 300, 8001)
t2 = np.linspace(300, 700, 8001)
t3 = np.linspace(700,1100, 8001)
t4 = np.linspace(1100,1500, 8001)
b_d1 = d1_ceu.ravel()
b_d2 = d2_ceu.ravel()
b_d3 = d3_ceu.ravel()
b_d4 = d4_ceu.ravel()


plt.figure()

plt.plot(t1,b_d1)
plt.plot(t2, b_d2)
plt.plot(t3, b_d3)
plt.plot(t4, b_d4)

plt.xlabel('time')
plt.ylabel('[D]')
plt.title('initial conc = 0.5')


plt.show()

