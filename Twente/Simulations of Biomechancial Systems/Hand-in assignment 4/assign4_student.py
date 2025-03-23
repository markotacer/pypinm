# -*- coding: utf-8 -*-
"""
Skeleton program for assignment 4. See assignment for detailed instructions. 
Make sure segdyn.py lives in the same directory as this file on your machine. 

@author: Koen Lemaire
"""
#%% imports
import segdyn
import scipy as sp
import scipy.integrate 
import numpy as np
import matplotlib.pyplot as plt
# this is ugly, but it is what it is for now. Check the imports so you know 
# which modules you have available with <%whos>
#%% segdynshell - a shell around segdyn to perform forward simulation
def segdynshell(t,state,parms):
    """
    segdynstated, Vnew = segdynshell_forward(t,segdynstate,segparms)
    
    Shell function around segdyn (see documentation in segdyn for details).
    segdynshell represents a set of ODE's that can be integrated with solve_ivp
    or similar. In segdynshell the dynamics of the 2D rigid body linkage are 
    defined through the vector V (see segdyn for details). Time dependent 
    inputs, external forces and constraints are also defined in this function.
    Note that for pure inverse applications where no simulation is required,
    segdyn should be callled directly. 
                          
    INPUT 
    t:        time (s)
    
    state:    vector containing segdynstate (see segdyn for definition) and any
              other state variables contained in the model
              
    parms:    must contain at least dict segparms with entries: 
              L=[l_1 .. l_n], d=[d_1 .. d_n], m=[m_1 .. m_n], and J=[J_1 .. J_n], where
              L_i is the length of the ith segment (m), d_i is the distance from the
              proximal end of the ith segment to its center of gravity (m), m_i is the
              mass of the ith segment (kg), and J_i the moment of inertia of the ith
              segment with respect to its center of gravity (kgm**2) 
              
    OUTPUT 

    stated:   first order derivative of state
    Vnew:     copy of V, with all unknown variables replaced with the value resulting
              from solving the equations of motion (see segdyn)
                 
   Author: Koen Lemaire (k.k.lemaire2@vu.nl)  
   """
    # read out parameters 
    K=parms['K']
    B=parms['B']
    segparms=parms['segparms']
    nseg=segparms['nseg']
    m=segparms['m']
    L=segparms['L']
    J=segparms['J']
    d=segparms['d']
    g=segparms['g']
   
    # Unravel state vector, partition in segdynstate and other states if needed
    M=state[-2:]
    segdynstate=state[0:-2] 
    phi=segdynstate[0:nseg]
    phid=segdynstate[nseg:2*nseg]
    xb=segdynstate[2*nseg:2*nseg+1]
    yb=segdynstate[2*nseg+1:2*nseg+2]
    xbd=segdynstate[2*nseg+2:2*nseg+3]
    ybd=segdynstate[2*nseg+3:2*nseg+4]
        
    # Application specific: external forces and moments
    #??
    
    # application specific: time variant input moments / forces
    # make stim:
    #??
    
    # calculate Md
    #??
    
    
    # Define V: this is where the basic dynamics are defined
    # Below we define the vector V, containing all (7*nseg+5) variables that could
    # potentially appear as unknowns in the equations of motion of a rigid body  
    # linkage (see definition and order of variables in comments behind each line).
    # Exactly 4*nseg+5 of these variables must contain an appropriate (known) 
    # value, the other 3*nseg variables must contain a np.nan (unknown) value.
    V=np.array(
        [np.nan, np.nan, 0, # Fx nseg+1 horizontal joint reaction forces
         np.nan, np.nan, 0, # Fy nseg+1 vertical joint reaction forces
         0, 0, 0,# M nseg+1 net joint moments
         0, 0,  # Fextx nseg horizontal external forces
         0, 0,  # Fexty nseg vertical externalforces
         0, 0, # Mext nseg external moments
         np.nan, np.nan, # phidd nseg segment angular accelerations
         0, 0], # [xbdd ybdd] horizontal and vertical base acceleration
         dtype=float) # make sure to cast as float (even when all input are int)
    
    # Application specific: constraints
    # to add constraints, first define the constraints at the postion level, 
    # then differentiate wrt time twice and construct Acons and Bcons such that
    # Acons@V=Bcons represents the twice differentiated constraint equations. 
    # Each contraint at the positinonal level corresponds to one row in Acons 
    # and Bcons. Thus Acons.shape=(n_constraints,7*nseg+5) and
    # Bcons.shape=(n_constraints,). After defining Acons and Bcons, make the 
    # appropriate values in V unknown (ie nan), such that:
    # n_constraints + 3*nseg = n_unknowns
    
    # define Acons and Bcons (assignment 4.6 and further):
    #Acons = ?? 
    #Bcons = ?? 
    
    # update V in case of contraints:
    # V[??]=??    
    
    # calculate segdynstated and Vnew:
    
    segdynstated, Vnew = segdyn.segdyn(segdynstate,segparms,V)  
    
    # Complete call to segdyn in case of constraints:
    #segdynstated, Vnew = segdyn.segdyn(segdynstate,segparms,V,Acons=Acons,Bcons=Bcons)  
    
    # define stated:
    stated=np.concatenate((segdynstated,Md)) # in case no other states present
    
    return stated, Vnew

#%% test segdyn
# define variable values
nseg=2
m=np.array([30, 50])
L=np.array([0.9, 0.8])
J=np.array([2, 3])
d=np.array([0.5, 0.3])
g=-9.81
K=200
B=25

# segment parameters dict
segparms = {'nseg':nseg, # number of segments
            'm':m, # mass of each segment [kg]
            'L':L, # length of each segment [m]           
            'd':d, # distance of COM of segment from proximal joint [m]
            'J':J, # moment of inertia about COM of segment [kgm**2]
            'g':g} # gravitational acceleration [m/s**2]

parms = {'segparms': segparms,
         'K':K,
         'B':B}


# define initial condition:
phi=np.array([np.pi/3, np.pi/6])
phid=np.array([0, 0])
# define segdynstate at t=0
segdynstate0=np.concatenate((phi, phid, np.zeros(4)))
# total system state at t=0
state0=segdynstate0


# test segdynshell
stated0, Vnew = segdynshell(0, state0, parms)
print(Vnew)
print(stated0)

# do simulation
t_span=[0, 4]
odefun= lambda t,state: segdynshell(t, state, parms)[0]
sol=sp.integrate.solve_ivp(odefun,t_span,state0,rtol=1e-8,atol=1e-8)
t=sol.t
state=sol.y
segdynstate=state[:-2,:]
plt.figure()
plt.subplot(1,2,1)
plt.plot(sol.t,state[0,:],'k')
plt.subplot(1,2,2)
plt.plot(sol.t,state[1,:],'k')

# calculate segment energies
Ekinx, Ekiny, Erot, Epot, Etot = segdyn.energy(segdynstate,segparms)

# calculate other stuff (linear / angular momenten, whatever, check out segdyn!)??

# some plots
# start with clean sheet
plt.close('all')

# plot states
plt.figure()
plt.subplot(1,2,1)
plt.plot(t,state[0,:],'k')
plt.xlabel('Time [s]')
plt.ylabel('$\phi_1$ [rad]')

plt.subplot(1,2,2)
plt.plot(t,state[1,:],'k')
plt.xlabel('Time [s]')
plt.ylabel('$\phi_2$ [rad]')


# plot total energy
plt.figure()
plt.plot(sol.t,Etot,'k')

# animation
anim=segdyn.animate(t,segdynstate,segparms)

# %%
