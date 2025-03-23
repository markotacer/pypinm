# -*- coding: utf-8 -*-
"""
File to run assignment 5. Each set of subquestions is associated with one cell.
Run / finish the cells one by one (don't run entire program b/c it will err...)

@author: Koen Lemaire
"""
#%% imports
import segdyn
import scipy as sp
import scipy.integrate 
import numpy as np
import matplotlib.pyplot as plt
import woodblock

#%% main program
# bullet properties:
m_b=np.array([0.01]) # [kg] mass of our bullet
v_b=np.array([0, 400]) # [m/s] velocity of bullet at impact

# block properties:
nseg=1
m=np.array([1.0]) # [kg] mass of wood block
L=np.array([0.2]) # [m] length of wood block
J=np.array([0.005]) # [kgm^2] moment of inertia about COM of wood block
d=np.array([0.1]) # [m] distance of COM wrt base of wood block
g=np.array([-9.81]) # [m/s^2] gravitational acceleration

# segment parameters
segparms = {'nseg':nseg, # number of segments
            'm':m, # mass of each segment [kg]
            'L':L, # length of each segment [m]           
            'd':d, # distance of COM of segment from proximal joint [m]
            'J':J, # moment of inertia about COM of segment [kgm**2]
            'g':g} # gravitational acceleration [m/s**2]

parms = {'segparms': segparms,
         'm_b':m_b,
         'v_b':v_b}

# these parameters are such that we more or less are going to match the video 
# data!  

#%% flight phase

segdynstate0=np.array([1, 0, 0, 1, 0, 0]) # phi0, phid0, base0, based0
t_span=[0, 5]

# integration
odefun=lambda t,state,parms: woodblock.flight(t, state, parms)[0]
sol=sp.integrate.solve_ivp(odefun,t_span,segdynstate0,args=(parms,),
                           events=woodblock.detect_landing,rtol=1e-8,atol=1e-8)
t=sol.t
state=sol.y.copy()
#%% plastic collision
# define state prior to collsion: state_min
# ??
# check which end has reached floor and set base_landing flag
# ??
# calculate delta_state
# ??
# add delta state to state_min 
# ??
# check that state after collision is as you expect
# ??
# calculate energy lost during collision / work done by impulsive force

#%% elastic collision
# define state prior to collsion: state_min
# ??
# check which end has reached floor and set base_landing flag
# ??
# calculate delta_state
# ??
# add delta state to state_min 
# ??
# check that state after collision is as you expect
# ??
# calculate energy lost during collision / work done by impulsive force

#%% bullet block experiment
