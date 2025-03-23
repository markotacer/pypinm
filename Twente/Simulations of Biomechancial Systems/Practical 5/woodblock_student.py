# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:55:33 2022

@author: kle206
"""
import numpy as np
import segdyn
#%% flight phase 
def flight(t,state,parms):
    """
    segdynstated, Vnew = segdynshell(t,segdynstate,segparms)
    
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
              
    segparms: dictionary with fields: 
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
    segparms=parms['segparms']
    nseg=segparms['nseg']
    m=segparms['m']
    L=segparms['L']
    J=segparms['J']
    d=segparms['d']
    g=segparms['g']
   # Unravel state vector
    segdynstate=state # only in case the model has no other state variables
    
    phi=segdynstate[0:nseg]
    phid=segdynstate[nseg:2*nseg]
    xb=segdynstate[2*nseg:2*nseg+1]
    yb=segdynstate[2*nseg+1:2*nseg+2]
    xbd=segdynstate[2*nseg+2:2*nseg+3]
    ybd=segdynstate[2*nseg+3:2*nseg+4]
        
    # Application specific: external forces and moments
    # any external forces here ??
    ??
    
    # Define V: this is where the basic dynamics are defined
    # Below we define the vector V, containing all (7*nseg+5) variables that could
    # potentially appear as unknowns in the equations of motion of a rigid body  
    # linkage (see definition and order of variables in comments behind each line).
    # Exactly 4*nseg+5 of these variables must contain an appropriate (known) 
    # value, the other 3*nseg variables must contain a np.nan (unknown) value.
    
    # we are flying!!
    V=np.array(
        [0, 0, # Fx nseg+1 horizontal joint reaction forces
         0, 0, # Fy nseg+1 vertical joint reaction forces
         0, 0, # M nseg+1 net joint moments
         0, # Fextx nseg horizontal external forces
         0, # Fexty nseg vertical externalforces
         0, # Mext nseg external moments
         0, # phidd nseg segment angular accelerations
         0, 0], # [xbdd ybdd] horizontal and vertical base acceleration
         dtype=float) # make sure to cast as float (even when all input are int)
    
    # Application specific: constraints
    # to add a constraint, first define the constraint at the postion level, 
    # then differentiate wrt time twice and construct Acons and Bcons such that
    # Acons@V=Bcons represents the twice differentiated constraint equation. 
    # Each contraint at the positinonal level corresponds to one row in Acons 
    # and Bcons. Thus Acons.shape=(n_constraints,7*nseg+5) and
    # Bcons.shape=(n_constraints,). After defining Acons and Bcons, make the 
    # appropriate values in V unknown (ie nan), such that:
    # n_constraints + 3*nseg = n_unknowns
    
    # define Acons and Bcons:
    # Acons =     
    # Bcons = 
    
    # update V in case of contraints:
    # V[..] =    
    
    # calculate segdynstated and Vnew:
    segdynstated, Vnew = segdyn.segdyn(segdynstate,segparms,V,Acons=None,Bcons=None)  
    
    # calculate other state derivative values (if needed):
        
    
    # define stated:
    stated=segdynstated # in case no other states present
    
    return stated, Vnew
#%% landing detection
def detect_landing(t,state,parms):
    '''
    criterion=dectect_landing(state,parms)
    Detects when woodblock lands
    '''
        
    segdynstate=state.copy() # only in case the model has no other state variables
    segparms=parms['segparms']
    # criterion?? (make sure it covers both cases)
    # some calculation??
    criterion='??'
    return criterion

# termination critera
detect_landing.terminal=True
detect_landing.direction=-1.

#%% plastic collision
def plastic(state_min,parms,base_landing):
    """
    segdynstated, Vnew = segdynshell(t,segdynstate,segparms)
    
    Shell function around segdyn (see documentation in segdyn for details).
    segdynshell represents a set of ODE's that can be integrated with solve_ivp
    or similar. In segdynshell the dynamics of the 2D rigid body linkage are 
    defined through the vector V (see segdyn for details). Time dependent 
    inputs, external forces and constraints are also defined in this function.
    Note that for pure inverse applications where no simulation is required,
    segdyn should be callled directly. 
                          
    INPUT 
    
    state:    vector containing segdynstate (see segdyn for definition) and any
              other state variables contained in the model
              
    segparms: dictionary with fields: 
              L=[l_1 .. l_n], d=[d_1 .. d_n], m=[m_1 .. m_n], and J=[J_1 .. J_n], where
              L_i is the length of the ith segment (m), d_i is the distance from the
              proximal end of the ith segment to its center of gravity (m), m_i is the
              mass of the ith segment (kg), and J_i the moment of inertia of the ith
              segment with respect to its center of gravity (kgm**2) 
    
    base_landing: Bolean flag indicating wether we land on base (True) or end (False)
              
    OUTPUT 

    stated:   first order derivative of state
    Vnew:     copy of V, with all unknown variables replaced with the value resulting
              from solving the equations of motion (see segdyn)
                 
   Author: Koen Lemaire (k.k.lemaire2@vu.nl)  
   """
    # read out parameters 
    segparms=parms['segparms']
    nseg=segparms['nseg']
    m=segparms['m']
    L=segparms['L']
    J=segparms['J']
    d=segparms['d']
    g=segparms['g']
   # Unravel state vector
   # YOU NEED TO DO COPY HERE!!! 
    segdynstate=state_min.copy() # only in case the model has no other state variables
    phi=segdynstate[0:nseg]
    phid=segdynstate[nseg:2*nseg]
    xb=segdynstate[2*nseg:2*nseg+1]
    yb=segdynstate[2*nseg+1:2*nseg+2]
    xbd=segdynstate[2*nseg+2:2*nseg+3]
    ybd=segdynstate[2*nseg+3:2*nseg+4]
        
    # Application specific: external forces and moments
    # doing collision!!??
    
    # application specific: time variant input moments / forces    
    
    # Define V: this is where the basic dynamics are defined
    # Below we define the vector V, containing all (7*nseg+5) variables that could
    # potentially appear as unknowns in the equations of motion of a rigid body  
    # linkage (see definition and order of variables in comments behind each line).
    # Exactly 4*nseg+5 of these variables must contain an appropriate (known) 
    # value, the other 3*nseg variables must contain a np.nan (unknown) value.
    
    # change appropriate variables here first, start with flight situation
    V=np.array(
        [0, 0, # Px nseg+1 horizontal joint reaction forces
         0, 0, # Py nseg+1 vertical joint reaction forces
         0, 0, # M nseg+1 net joint moments
         0, # Pextx nseg horizontal external forces
         0, # Pexty nseg vertical externalforces
         0, # Mext nseg external moments
         0, # detla phid nseg segment changes in angular velocity
         0, 0], # delta [xbd ybd] horizontal and vertical base changes in velocity
         dtype=float) # make sure to cast as float (even when all input are int)
    
    # Application specific: constraints
    # to add a constraint, first define the constraint at the postion level, 
    # then differentiate wrt time twice and construct Acons and Bcons such that
    # Acons@V=Bcons represents the twice differentiated constraint equation. 
    # Each contraint at the positinonal level corresponds to one row in Acons 
    # and Bcons. Thus Acons.shape=(n_constraints,7*nseg+5) and
    # Bcons.shape=(n_constraints,). After defining Acons and Bcons, make the 
    # appropriate values in V unknown (ie nan), such that:
    # n_constraints + 3*nseg = n_unknowns
    
    if base_landing: # we land on the base
        # what should now be knowns / unknowns? What are our constraint eqns??
        Acons = '??' 
        Bcons = '??'
        V['??']='??'         
    else: # we land on the the end
        Acons = '??' 
        Bcons = '??'         
        V['??']='??'         
        
    # Set finite terms in b_star to zero, done by setting angular velocities 
    # and base velocity to zero
    segdynstate[[1,-2,-1]]=0 # 
    
    
    # calculate segdynstated and Vnew:
    segdynstated, Vnew = segdyn.segdyn(segdynstate,segparms,V,Acons,Bcons)  
        
    
    # state after collision:
    delta_state=segdynstated
    
    return delta_state, Vnew

#%% elastic collision
def elastic(state_min,parms,base_landing):
    """
    segdynstated, Vnew = segdynshell(t,segdynstate,segparms)
    
    Shell function around segdyn (see documentation in segdyn for details).
    segdynshell represents a set of ODE's that can be integrated with solve_ivp
    or similar. In segdynshell the dynamics of the 2D rigid body linkage are 
    defined through the vector V (see segdyn for details). Time dependent 
    inputs, external forces and constraints are also defined in this function.
    Note that for pure inverse applications where no simulation is required,
    segdyn should be callled directly. 
                          
    INPUT 
    
    state:    vector containing segdynstate (see segdyn for definition) and any
              other state variables contained in the model
              
    segparms: dictionary with fields: 
              L=[l_1 .. l_n], d=[d_1 .. d_n], m=[m_1 .. m_n], and J=[J_1 .. J_n], where
              L_i is the length of the ith segment (m), d_i is the distance from the
              proximal end of the ith segment to its center of gravity (m), m_i is the
              mass of the ith segment (kg), and J_i the moment of inertia of the ith
              segment with respect to its center of gravity (kgm**2) 
    
    base_landing: Bolean flag indicating wether we land on base (True) or end (False)
              
    OUTPUT 

    stated:   first order derivative of state
    Vnew:     copy of V, with all unknown variables replaced with the value resulting
              from solving the equations of motion (see segdyn)
                 
   Author: Koen Lemaire (k.k.lemaire2@vu.nl)  
   """
    # read out parameters 
    segparms=parms['segparms']
    nseg=segparms['nseg']
    m=segparms['m']
    L=segparms['L']
    J=segparms['J']
    d=segparms['d']
    g=segparms['g']
   # Unravel state vector
    segdynstate=state_min.copy() # only in case the model has no other state variables
    phi=segdynstate[0:nseg]
    phid=segdynstate[nseg:2*nseg]
    xb=segdynstate[2*nseg:2*nseg+1]
    yb=segdynstate[2*nseg+1:2*nseg+2]
    xbd=segdynstate[2*nseg+2:2*nseg+3]
    ybd=segdynstate[2*nseg+3:2*nseg+4]
        
    # Application specific: external forces and moments
    # collision mechanics!!     
    
    # Define V: this is where the basic dynamics are defined
    # Below we define the vector V, containing all (7*nseg+5) variables that could
    # potentially appear as unknowns in the equations of motion of a rigid body  
    # linkage (see definition and order of variables in comments behind each line).
    # Exactly 4*nseg+5 of these variables must contain an appropriate (known) 
    # value, the other 3*nseg variables must contain a np.nan (unknown) value.
    
    # change appropriate variables here first, start with flight situation
    V=np.array(
        [0, 0, # Px nseg+1 horizontal joint reaction forces
         0, 0, # Py nseg+1 vertical joint reaction forces
         0, 0, # M nseg+1 net joint moments
         0, # Pextx nseg horizontal external forces
         0, # Pexty nseg vertical externalforces
         0, # Mext nseg external moments
         0, # detla phid nseg segment changes in angular velocity
         0, 0], # delta [xbd ybd] horizontal and vertical base changes in velocity
         dtype=float) # make sure to cast as float (even when all input are int)
    
    # Application specific: constraints
    # to add a constraint, first define the constraint at the postion level, 
    # then differentiate wrt time twice and construct Acons and Bcons such that
    # Acons@V=Bcons represents the twice differentiated constraint equation. 
    # Each contraint at the positinonal level corresponds to one row in Acons 
    # and Bcons. Thus Acons.shape=(n_constraints,7*nseg+5) and
    # Bcons.shape=(n_constraints,). After defining Acons and Bcons, make the 
    # appropriate values in V unknown (ie nan), such that:
    # n_constraints + 3*nseg = n_unknowns
    
    if base_landing: # we land on the base
        # what should now be knowns / unknowns? What are our constraint eqns??
        Acons = '??' 
        Bcons = '??'
        V['??']='??'         
    else: # we land on the the end
        Acons = '??' 
        Bcons = '??'         
        V['??']='??'         
    
    
    # Set remaining terms in b_star to zero, done by setting angular velocities 
    # and base velocity to zero
    segdynstate[[1,-2,-1]]=0 # 
    
    # calculate segdynstated and Vnew:
    segdynstated, Vnew = segdyn.segdyn(segdynstate,segparms,V,Acons,Bcons)  
    
    # state after collision:
    delta_state=segdynstated 
    
    return delta_state, Vnew

#%% bullet impact!
def bullet(state_min,parms,base_landing):
    """
    segdynstated, Vnew = segdynshell(t,segdynstate,segparms)
    
    Shell function around segdyn (see documentation in segdyn for details).
    segdynshell represents a set of ODE's that can be integrated with solve_ivp
    or similar. In segdynshell the dynamics of the 2D rigid body linkage are 
    defined through the vector V (see segdyn for details). Time dependent 
    inputs, external forces and constraints are also defined in this function.
    Note that for pure inverse applications where no simulation is required,
    segdyn should be callled directly. 
                          
    INPUT 
    
    state:    vector containing segdynstate (see segdyn for definition) and any
              other state variables contained in the model
              
    segparms: dictionary with fields: 
              L=[l_1 .. l_n], d=[d_1 .. d_n], m=[m_1 .. m_n], and J=[J_1 .. J_n], where
              L_i is the length of the ith segment (m), d_i is the distance from the
              proximal end of the ith segment to its center of gravity (m), m_i is the
              mass of the ith segment (kg), and J_i the moment of inertia of the ith
              segment with respect to its center of gravity (kgm**2) 
    
    base_landing: Bolean flag indicating wether we land on base (True) or end (False)
              
    OUTPUT 

    stated:   first order derivative of state
    Vnew:     copy of V, with all unknown variables replaced with the value resulting
              from solving the equations of motion (see segdyn)
                 
   Author: Koen Lemaire (k.k.lemaire2@vu.nl)  
   """
    # read out parameters 
    segparms=parms['segparms']
    nseg=segparms['nseg']
    m=segparms['m']
    L=segparms['L']
    J=segparms['J']
    d=segparms['d']
    g=segparms['g']
    
    # bullet parameters (we need them here!)
    m_b=parms['m_b']
    v_b=parms['v_b']
    
    # Unravel state vector
    segdynstate=state_min.copy() # only in case the model has no other state variables
    phi=segdynstate[0:nseg]
    phid=segdynstate[nseg:2*nseg]
    xb=segdynstate[2*nseg:2*nseg+1]
    yb=segdynstate[2*nseg+1:2*nseg+2]
    xbd=segdynstate[2*nseg+2:2*nseg+3]
    ybd=segdynstate[2*nseg+3:2*nseg+4]
        
    # Application specific: external forces and moments
    # collision mechanics!!     
    
    # Define V: this is where the basic dynamics are defined
    # Below we define the vector V, containing all (7*nseg+5) variables that could
    # potentially appear as unknowns in the equations of motion of a rigid body  
    # linkage (see definition and order of variables in comments behind each line).
    # Exactly 4*nseg+5 of these variables must contain an appropriate (known) 
    # value, the other 3*nseg variables must contain a np.nan (unknown) value.
    
    # change appropriate variables here first, start with flight situation
    V=np.array(
        [0, 0, # Px nseg+1 horizontal joint reaction forces
         0, 0, # Py nseg+1 vertical joint reaction forces
         0, 0, # M nseg+1 net joint moments
         0, # Pextx nseg horizontal external forces
         0, # Pexty nseg vertical externalforces
         0, # Mext nseg external moments
         0, # detla phid nseg segment changes in angular velocity
         0, 0], # delta [xbd ybd] horizontal and vertical base changes in velocity
         dtype=float) # make sure to cast as float (even when all input are int)
    
    # Application specific: constraints
    # to add a constraint, first define the constraint at the postion level, 
    # then differentiate wrt time twice and construct Acons and Bcons such that
    # Acons@V=Bcons represents the twice differentiated constraint equation. 
    # Each contraint at the positinonal level corresponds to one row in Acons 
    # and Bcons. Thus Acons.shape=(n_constraints,7*nseg+5) and
    # Bcons.shape=(n_constraints,). After defining Acons and Bcons, make the 
    # appropriate values in V unknown (ie nan), such that:
    # n_constraints + 3*nseg = n_unknowns
    
    # bullet impact, what should now be the knowns/unknowns?? How is (change in)
    # bullet velocity related to the impulses? 
    Acons = '??' 
    Bcons = '??'
    V['??']='??'         
    
    # Set remaining terms in b_star to zero, done by setting angular velocities 
    # and base velocity to zero
    segdynstate[[1,-2,-1]]=0 # 
    
    # calculate segdynstated and Vnew:
    segdynstated, Vnew = segdyn.segdyn(segdynstate,segparms,V,Acons,Bcons)  
    
    # state after collision:
    delta_state=segdynstated 
    
    return delta_state, Vnew