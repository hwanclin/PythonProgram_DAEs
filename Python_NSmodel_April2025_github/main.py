#Source File: main.py

import numpy as np
from   NSmodel import *
from   scipy.optimize import fsolve, minimize_scalar
from   scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import pandas as pd

#===============================================================================
# I. INTRODUCTION
#
# The Python program includes two source files (main.py & NSmodel.py):
#
# 1. main.py -- This source file includes the driver called "main()" and nine 
# implementation functions designed to compute a dynamical system's steady
# state, transitional dynamics, intertemporal welfare, among other things.
# 
# 2. NCmodel.py -- This source file defines a class called "NSmodel". The
# class includes "class members" and "class functions" used to:
# 
# a) Import the values of benchmarked  parameters for a North-South model
#    developed by Professor Hwan C. Lin of UNC-Charlotte;   
# b) Change some of these parameter values for IPRs-strengthening experiments;
# c) Define the North-South model's reduced-form steady-state system;
# d) Compute the steady-state equilibrium values of some endogenous variables;
# e) Define the North-South model's dynamical system that evolves over time 
#    in transition to steady state in the semi-infinite time domain [0,inf); and
# f) Define a set of two-point boundary conditions for the dynamical system.
#
# II. The North-South Dynamical System
#
# Equations (1)-(3) define a three-dimensional dynamical system below for the 
# North-South model,
#
#    dzetax(t)/dt = F1(Y(t), Z(t))                                       (1)
#
#    dzetay(t)/dt = F2(Y(t), Z(t))                                       (2)
#
#    dGamma(t)/dt = F1(Y(t), Z(t))                                       (3)
#       
#    s.t. the following five static-equilibrium constraints at any moment:
#
#    Gi(Y(t),Z(t)) = 0, i = {1,2,3,4,5}, t in [0,inf)                    (4) 
#
# where Y(t) = [zetax(t), zetay(t), Gamma(t)]
#       Z(t) = [gx(t), gy(t), thetax(t), thetay(t), tau(t)]
#
# Note that Given Y(t) at any point in time, the five-dimensional vector Z(t)
# can be determined by solving the five static-equilibrium constraints given
# in (4). As such, the North-South dynamical system is a three-dimensional 
# system of Differential Algebraic Equations (DAEs), in contrast to Ordinary
# Differential Equations (ODEs).
#   
# The entire dynamical system of DAEs is laid out in equations (22)-(29) of
# Professor Lin's paper titled "DAEs-driven dynamics and North-South protection
# of IPRs." This paper now is titled "A dynamic modelling approach to 
# North-South disparities in IPR protection" and published in Oxford 
# Economic Papers [https://doi.org/10.1093/oep/gpaf024].
#
# III. The two-point boundary conditions
#
# The dynamical system of (1)-(3) subject to the five constraints in (4) 
# presents a boundary value problem (BVP) in that both zetax(t) and zetay(t)
# are predetermined state variables at any point in time while Gamma(t) is a
# "jump" variable at any moment to respond to any shock coming to perturb the
# North-South model. Thus, whenever the standard of IPRs is strengthened
# initially at t=0, for instance, in the South, to solve the constrained
# dynamical system requires a set of two-point boundary conditions as given 
# below,
#   
#    zetax(t) = zetax0 at t=0
#    zetay(t) = zetay0 at t=0
#    Gamma(t) = Gamma1 at t->inf
#
# IV. Python's BVP solver and the iteration mechanism
#
# Any BVP solver must involves some iteration mechanisms until the solution
# converges to a predetermined error tolerance. 
#
# We use Python's BVP solver -- scipy.optimize.solve_bvp -- to solve the
# dynamical system of DAEs as a boundary value problem associated with a
# constrained system of ODEs. Thus, in the iteration process, we must solve the 
# five constraints of (4) for Z(t) at each node in the time mesh before its
# inner mechanism can work to solve the dynamical system iteratively.
#
# V. Time paths of endogenous variables and intertemporal welfare
#
# Once the dynamical system is solved successfully, we can obtained the time
# paths of some endogenous variables to compute the change in intertemporal
# welfare for the North and South using the formulas provided in Propositions 4
# and 5 of Professor Lin's paper.
#
# VI. Robustness Checks
#
# The Python program includes an implementaion function called
# "robustness_checks(.....), which is designed to examine how the intertemporal 
# welfare change resulting from a tightening of Southern IPRs is sensitive to
# the model parameter of creative destruction. As such robustness checks
# involve fifteen scenarios, one should expect a bit more time (like 2 to 3 
# minutes) required to see results showing up on the console.
#
# VII. How to test the Python program?
#
#       Step 1: Place the two source files (main.py & NCmodel.py) in the same 
#               working directory or folder of your local computer.
#
#       Step 2: Open the driver file of main.py.
#
#       Step 3: Run the driver (main.py) and you will see all the results that 
#               appear in the paper by Hwan C. Lin.
#     
# Python Coder: Professor Hwan C. Lin                    Date: April 15, 2025   
#===============================================================================

def main():
    """
    Three tasks are done here:
    1) Compute steady states for four scenarios distinguished by different
       imitation rates of good y, which is exclusively for Southern consumers.
    2) Solve the dynamical system of DAEs and plot the time paths of some
       variables of interest.
    3) Compute impacts on both steady-state welfare and intertemporal welfare. 
    """
    #---------------------------------------------------------------
    # Baseline Model Parameters
    #--------------------------------------------------------------- 
    LN  = 1.0                   #Labor force in North 
    LS  = 2.0                   #Labor force in South
    a   = 1.5                   #Research productivity
    rho = 0.025                 #Time preference
    mx  = 0.02                  #Imitation rate of good x
    my  = 0.10                  #Imitation rate of good y
    epsilon = 2.0               #Elasticity of demand
    eta = epsilon/(epsilon-1.0) #Markup on unimitated goods
    psi = 1.0                   #Coefficient of creative destruction
    #---------------------------------------------------------------
    

    #-------------------------------------------------------------------
    # Initiate a class object called "dydt" associated with the class of 
    # "NSmodel" declared in the source file, NSmodel.py.
    #-------------------------------------------------------------------
    
    dydt = NSmodel(LN, LS, a, epsilon, rho, mx, my, eta, psi)

    #------------------------------------------------------------------- 
    # Note: By initiating the class object, the nine model parameters 
    #       are exported to the class as class members, which are used
    #       to parametrize all class functions including the North-South
    #       dynamical system defined in the class. Further, any function
    #       outside of the class can access all class memebers and class 
    #       functions though the class object, dydt.
    #-------------------------------------------------------------------

    
    #---------------------------------------------------------------------
    # Compute 4 Steady States for my=0.02, 0.05, 0.10, 0.20, respectively.
    #---------------------------------------------------------------------
    #Initial guess
    x0 = [0.10, 0.10, 0.5, 0.5, 1.5, 0.5, 0.5]
    # where x0 is the solution guess of                                      
    # [gx, gy, zetax, zetay, tau, thetaN, thetaS] that solves the steady-state
    # system,                                                                  
               
    #Parameter range for y-good imitation rate (my)  
    my_choice = [0.02, 0.05, 0.10, 0.20]

    #Parameter range for creative-destruction coefficient (psi)
    psi_choice = [0.50, 0.75, 1.00, 1.25, 1.30]
    
    #Remarks: psi is set equal to the benchmark value (psi=1.00) for the 
    #computation of 4 steady states (subject to my) in the for-loop:

    for i in range(4):

        #Parametrize the model using each of the 4 y-good imitation rates
        dydt.setmy(my_choice[i])
        
        #Call function ssEquilibrium(...) to obtain steady state  
        SteadyState = ssEquilibrium(dydt, mx, my_choice[i], psi, x0)

        if i==0:
            ss_Scenarios = SteadyState
        else:
            ss_Scenarios = np.vstack((ss_Scenarios, SteadyState)) 
        #if i==2: SteadyState_0 = SteadyState
        
        #Remarks: As i = 0, mx = 0.02 & my = 0.02
        #         As i = 1, mx = 0.02 & my = 0.05
        #         As i = 2, mx = 0.02 & my = 0.10 (benchmark scenario)
        #         As i = 3, mx = 0.02 & my = 0.20

    #Display the 4 steady states
    displaySS(ss_Scenarios)  
    #---------------------------------------------------------------------


    #---------------------------------------------------------------------
    # Compute steady-state welfare effects
    #---------------------------------------------------------------------
    # Retrieve new steady state with my matched to mx
    SteadyState_1 = ss_Scenarios[0][2:10]

    for i in range(1,4):

        #Retrieve initial steady state with my < mx
        SteadyState_0 = ss_Scenarios[i][2:10]

        N_Omega_ss, S_Omega_ss = dydt.Steady_Omega(SteadyState_0, SteadyState_1)

        #where
        #   N_Omega_ss is of size 5 for the North::
        #   N_Omega_ss[0]: Steady-state welfare change, measured by Omega;
        #   N_Omega_ss[1]: Product-innovation effect;
        #   N_Omega_ss[2]: Terms of trade effect;
        #   N_Omega_ss[3]: Market_power effect; 
        #   N_Omega_ss[4]: Saving effect.
        #
        #   S_Omega_ss is of size 5 for the South:
        #   S_Omega_ss[0]: Steady-state welfare change, measured by Omega;
        #   S_Omega_ss[1]: Product-innovation effect;
        #   S_Omega_ss[2]: Terms of trade effect;
        #   S_Omega_ss[3]: Market_power effect;
        #   S_Omega_ss[4]: Saving effect, which is None for the South.
        
        if i==1: 
            N_Omega_ss_all = N_Omega_ss
            S_Omega_ss_all = S_Omega_ss
        else:
            N_Omega_ss_all = np.vstack((N_Omega_ss_all, N_Omega_ss))
            S_Omega_ss_all = np.vstack((S_Omega_ss_all, S_Omega_ss))    
        
    #Display steady-state welfare effects
    displaySS_welfare(my_choice, N_Omega_ss_all, S_Omega_ss_all)
    #-------------------------------------------------------------------

    
    #--------------------------------------------------------------------------
    # Solve the North-South dynamical system as a constrained BVP:
    #   1) The dynamical system is a set of DAEs in the time domain [0, inf).
    #   2) The DAEs are solved as a 3-dimensional constrained system of ODEs.
    #   3) The constrained system of ODEs is subject to 5 constraints.
    #   4) The constrained system of ODEs is for the 3 endogenous variables:
    #      -> zetax, the unimitated fraction of goods consumed only in North
    #      -> zetay, the unimitated fraction of goods consumed only in South
    #      -> Gamma, the sum of x-good innovation rate (gx) and
    #                y-good innovation rate (gy)
    #   5) The constrained system of ODEs is defined in the class, NCmodel.
    #   6) Python's BVP solver (scipy.integrate.solve_bvp) is used in the
    #      used-defined function, solveDySystem(...), provided below.
    #--------------------------------------------------------------------------

    #Solution holders for three experiments in the following for-loop
    solall0 = []  #For psi[0] = 0.50 
    solall1 = []  #For psi[1] = 0.75
    solall2 = []  #For psi[2] = 1.00
    solall3 = []  #For psi[3] = 1.25
    solall4 = []  #For psi[4] = 1.30

    #---------------------------------------------------------------------- 
    # The two for-loops below are nested to simulate 15 scenarios due to
    # three policy experiments for each of the five parameter values of psi
    # selected from the  psi_choice list of size 5. 
    # ---------------------------------------------------------------------

    my1 = my_choice[0]
    
    for i in range(5):
        
        psi = psi_choice[i] 

        for j in range(1,4):
            """
            j=1: Experiment 1 - my is decreased from 0.05 to 0.02 = mx
            j=2: Experiment 2 - my is decreased from 0.10 to 0.02 = mx
            j=3: Experiment 3 - my is decreased from 0.20 to 0.02 = mx
            """

            my0 = my_choice[j]
         
            time_max   = 250   #chosen to be an approximation of inf in time
            time_nodes = 100   #nodes in in the time mesh [0,..., time-max] 
            tol        = 1e-6  #error tolerance

            sol, Old_SS, New_SS = solveDySystem (  dydt,       
                                                   mx,        
                                                   my0,        
                                                   my1,      
                                                   psi,        
                                                   x0,         
                                                   time_max,  
                                                   time_nodes, 
                                                   tol )

            #-------------------------------------------------------------- 
            # Calling solveDySystem(...) can obtain three outputs:
            #
            #   sol -- the solution for the dynamical system, dydt, 
            #          for a specific experiment with my raised to match mx,
            #          given a specific value of psi (creative destruction).
            #
            #   Old_SS -- The dynamical system's initial steady state.
            #
            #   New_SS -- The dynamical system's new steady state.
            #--------------------------------------------------------------- 
            

            # Collect results from each of the 15 scenarios:

            if i==0:
                solall0.append(sol)
                if j==1:
                    Old_SS0 = Old_SS
                    New_SS0 = New_SS
                else:
                    Old_SS0 = np.vstack((Old_SS0, Old_SS))
                    New_SS0 = np.vstack((New_SS0, New_SS))

            elif i==1:
                solall1.append(sol)
                if j==1:
                    Old_SS1 = Old_SS
                    New_SS1 = New_SS
                else:
                    Old_SS1 = np.vstack((Old_SS1, Old_SS))
                    New_SS1 = np.vstack((New_SS1, New_SS))
            elif i==2:
                solall2.append(sol)
                if j==1:
                    Old_SS2 = Old_SS
                    New_SS2 = New_SS
                else:
                    Old_SS2 = np.vstack((Old_SS2, Old_SS))
                    New_SS2 = np.vstack((New_SS2, New_SS))

            elif i==3:
                solall3.append(sol)
                if j==1:
                    Old_SS3 = Old_SS
                    New_SS3 = New_SS
                else:
                    Old_SS3 = np.vstack((Old_SS3, Old_SS))
                    New_SS3 = np.vstack((New_SS3, New_SS))

            else:
                solall4.append(sol)
                if j==1:
                    Old_SS4 = Old_SS
                    New_SS4 = New_SS
                else:
                    Old_SS4 = np.vstack((Old_SS4, Old_SS))
                    New_SS4 = np.vstack((New_SS4, New_SS))
            
    #--------------------------------------------------------
    # From the above nested for-loops and if-controls, 
    # we can obtain the following three soluion holders:
    # -------------------------------------------------------
    
    solall = [solall0, solall1, solall2, solall3, solall4]
    Old_SS = [Old_SS0, Old_SS1, Old_SS2, Old_SS3, Old_SS4]
    New_SS = [New_SS0, New_SS1, New_SS2, New_SS3, New_SS4]

    #-----------------------------------------------------------------
    # Remarks: solall0, solall1, solall2, solall3, and 
    #          solall4  hold information about the time
    #          paths of zetax(t), zetay(t), and Gamma(t).
    #
    # Of these solution holders:
    #
    #   solall0 -- holds the solution of zeta(t), zetay(t),
    #              and Gamma(t) from Experiments 1, 2, and 3,
    #              given  psi = psi_choice[0].
    #
    #   solall1 -- holds such information, given psi = psi_choice[1].
    #
    #   solall2 -- holds such information, given psi = psi_choice[2].
    #
    #   solall3 -- holds such information, given psi = psi_choice[3].
    #
    #   solall4 -- holds such information, given psi = psi_choice[4].
    #
    #  In addition, the above nested for-loop and if-controls can generate 
    #  Initial_SS0, ..., Initial_SS4 and New_SS0, ..., New_SS4,
    #  which hold the same dynamical system's intial and new steady
    #  states for each of Experiments 1, 2, and 3, given each value 
    #  of psi in the psi_choice list of size 5..
    #------------------------------------------------------------------

    

    #---------------------------------------------------------------------------
    # Plot the time paths in transition to steady state for the benchmark
    # experiment, for which the initial y-good imitation rate is set equal to
    # my0=0.10 and raised to my1 in order to match the x-good imitation rate,
    # denoted by mx=0.02. 
    # 
    # In addition, we can use the time paths of zetax(t), zetay(t), and Gamma(t)
    # to plot the dynamical system's "stable manifold" in a 3D space.
    #---------------------------------------------------------------------------

    i = 2   #With this i-index, psi = psi_choice[2] = 1.00, the benchmark value
    j = 1   #With this j-index, solall[1], Inital_SS2[1], and New_SS2[1] all
            #refer to the benchmark experiment, given psi=1.00.

    # Time paths in transition
    plot_timepaths(dydt, solall, Old_SS, New_SS, psi_choice, i, j)

    # Stable Manifold in transition
    plot_stable_manifold(dydt, solall, Old_SS, New_SS, psi_choice, i, j)

    #------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    # Compute and display intertemporal-welfare effects
    #--------------------------------------------------------------------------
    
    i = 2

    for j in range(3):
        
        N_Omega_Transition, S_Omega_Transition =\
        intertemporal_welfare(dydt, solall, Old_SS, New_SS, psi_choice, i, j,\
                              my_choice)

        displayOmega_Transition(N_Omega_Transition, S_Omega_Transition, \
                                psi_choice, i, j, my_choice)

    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------
    # Robustness Checks on psi, coefficient of creative destruction
    #--------------------------------------------------------------------

    robustness_checks(dydt,solall,Old_SS,New_SS,psi_choice,i,j,my_choice)
    
    #-------------------------------------------------------------------- 
    # Remarks: Robustness checks are aimed to see if the intertemporal
    #          welfare changes are sensitive to the creative destruction 
    #          coefficient psi. 
    #--------------------------------------------------------------------

    
    #           ****** End of the file of main.py *****


#============================================================================
# Declared below are Implementation Functions callable in the driver main().
# They are: 
# 1) ssEquilibrium(-----)
# 2) solveDySystem(-----)
# 3) timepaths(-----)
# 4) plot_timepaths(-----)
# 5) intertemporal_welfare(-----)
# 6) displaySS(-----)
# 7) displaySS_welfare(-----)
# 8) displayOmega_Transition(-----)
# 9) robustness_checks(-----)  
#============================================================================

def ssEquilibrium(dydt, mx, my, psi, sol_guess):
    """
    Input
      dydt - the class object.
      mx   - the x-good imitation rate.
      my   - the y-good imitation rate.
      psi  - the coefficient of creative destruction
      sol_guess - an array of 7 unknowns for solving the dynamical system's 
                steady state.
    Output - an array of size 13, including  mx, mx, and the following:
      Gamma  - the sum of gx and gy,
      gx     - innovation rate of good x
      gy     - innovation rate of good y
      zetax  - the unimitated fraction of good x                             
      zetay  - the unimitated fraction of good y                             
      tau    - North-South terms of trade applied to both good x and good y              
      thetaN - the share of Northern manufacturing labor employed to         
               produce good x                                                
      thetaS - the share of Southern manufacturing labor employed to         
               produce good y   
      Tx     - Expected patent life for good x
      Ty,    - Expected patent life for good y
      wNwS   - Northern wage relative to Southern wage

    """

    dydt.setmx(mx)
    dydt.setmy(my)
    dydt.setpsi(psi)

    SteadyState = fsolve(dydt.steady_state, sol_guess)
    
    #-----------------------------------------------------------------------
    # Remarks: SteadyState is an array of size 7, including 
    # gx, gy, zetax, zetay, tau, thetaN, thetaS.
    #-----------------------------------------------------------------------

    GammaSS = SteadyState[0]+SteadyState[1]  #SteadyState[0] = gx
                                             #SteadyState[1] = gy

    HazardRateSS_x = psi*SteadyState[0] + mx
    HazardRateSS_y = psi*SteadyState[1] + my

    Tx = 1/HazardRateSS_x     #Expected monopoly duration in x-industry
    Ty = 1/HazardRateSS_y     #expected monopoly duration in y industry
    eta = dydt.eta
    wNwS = SteadyState[4]/eta #North wage relative to Southern wage
                              #SteadyState[4] = tau or N-S terms of trade
    
    # Insert or append other endogenous variables to expand the size of 
    # SteadyState from 7 to 13: 

    SteadyState = np.insert(SteadyState, 0, GammaSS)
    SteadyState = np.insert(SteadyState, 0, my)
    SteadyState = np.insert(SteadyState, 0, mx)
    SteadyState = np.append(SteadyState, Tx)
    SteadyState = np.append(SteadyState, Ty)
    SteadyState = np.append(SteadyState, wNwS)

    return SteadyState   

#===============================================================================

def solveDySystem(dydt, mx, my0, my1, psi, sol_guess, tmax, tnodes, tol): 
    """
    Input
      dydt - the class object, which can refer to the dynamical system defined 
             in the class called  "NSmodel" in the source file of NSmodel.py.
      mx   - imitation rate of good x.
      my0  - initial imitation rate of good y at time t=0.
      my1  - new imitation rate of good y, set to take effect at time t=0.
      psi  - coefficient of creative destruction.
      sol_guess - an array of size 7, used for solving the steady state system.
      tmax   - the upper bound of time domain, used as an approximation of inf.
      tnodes - the number of nodes in the time mesh.
      tol    - error tolerance
    Output
      solution   - a solution bundle obtained by solving the dynamical system. 
      Initial_SS - the initial steady state before my is decreased 
                   from my0 to my1 at t=0.
      New_SS     - the new steady state after my is decreased 
                   from my0 to my1 at t=0.
    """

    # Parametrize the dynamical system using psi and my0 together with the other 
    # model parameters that has been exported to the class NCmodel.
    
    dydt.setpsi(psi)
    
    dydt.setmy(my0)
    Initial_SS = ssEquilibrium(dydt, mx, my0, psi, sol_guess)

    dydt.setmy(my1)
    New_SS = ssEquilibrium(dydt, mx, my1, psi, sol_guess)

    # Retrieve 3 individual variables' steady-state equilibrium values,  
    # respectively, from initial and new steady states:

    Gamma0, zetax0, zetay0 = Initial_SS[[2,5,6]]
    Gamma1, zetax1, zetay1 = New_SS[[2,5,6]]

    ssy0 = np.array([zetax0, zetay0, Gamma0]) #Initial steady state
    ssy1 = np.array([zetax1, zetay1, Gamma1]) #New steady state
    
    #Export ssy0 and ssy1 to the class to form two-point boundary conditions:

    dydt.setssy0(ssy0)
    dydt.setssy1(ssy1)
    bc = dydt.bc       #Two-point boundary conditions defined in the class.

    # Time mesh
    tmesh = np.linspace(0, tmax, tnodes)

    # Initial guess for the time paths of the system of DAEs
    zetax_guess = np.ones(tnodes)*(zetax0+zetax1)*0.5
    zetay_guess = np.ones(tnodes)*(zetay0+zetay1)*0.5
    Gamma_guess = np.ones(tnodes)*(Gamma0+Gamma1)*0.5

    y_guess = np.vstack((zetax_guess, zetay_guess, Gamma_guess))

    #Call scipy.integrate.solve_bvp to solve dynamical system, denoted by dydt
    solution = solve_bvp( dydt,
                          bc,
                          tmesh,
                          y_guess,
                          tol=tol,
                          max_nodes=1000,
                          verbose=0  )

    #--------------------------------------------------------------------------
    # Remarks:
    # 1) To see the information about the BVP-solving process,
    #    one can set "verbose=1" or "verbose=2" in the above.
    #      
    # 2) One can also unmark the marked code below to see more information:
    #print("\nFor the Experiment with psi=%3.2f, my0 = %3.2f and my1 = %3.2f" \
    #      % (psi, my0, my1))
    #print("success is: ", solution.success)
    #--------------------------------------------------------------------------

    return solution, Initial_SS, New_SS

    # Note: As noted above, the "solution" is a solution bundle, To understand
    # how to retrieve items from the solution bundle, see the Python website:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html

#===============================================================================

def timepaths(dydt, solall, psi_choice, i, j):
    """
    Input
      dydt       - the class object.
      solall     - the solution bundle for the dynamical system
      psi_choice - a list of size 5.
      i          - an index with psi_choice[i] referring to a specific psi.
      j          - an index referring to a specific experiment.
    Output - a tuple of size 9 including the time paths of 9 variables:
      zeta_x(t)  - the time path of zetax,
      zeta_y(t)  - the time path of zetay. 
      Gamma(t)   - the time path of Gamma. (Gamma = gx + gy)
      gamma(t)   - the time path of gamma. (gamma = gx - gy)
      g_x(t)     - the time path of gx.
      g_y(t)     , the time path of gy.
      tau(t)     - the time path of tau.   (North-South terms of trade)
      thetaN(t)  - the time path of thetaN.
      thetaS(t)  - the time path of thetaS. 
    """
    dydt.setpsi(psi_choice[i])  #i=2 refers to psi=1.00                         
                                                                                
    solution = solall[i]                                                        
                                                                                
    sol = solution[j].sol       #j=1 refers to Experiment2 : my0=0.10 --> 0.02.
                                #Experiment 2 is the benchmark experiment. 

    #---------------------------------------------------                        
    #The dynamical system evolves over time in terms of:                        
    def zeta_x(t):                                                              
        return sol(t)[0]                                                        
                                                                                
    def zeta_y(t):                                                              
        return sol(t)[1]                                                        
                                                                                
    def Gamma(t):                                                               
        return sol(t)[2]                                 
    #---------------------------------------------------                        
    #Use the above-extracted solution to construct the
    #time paths of some other endogenous variables.
    def tau(t):

        if np.size(t)==1:

            tau0 = 2.0

        else:

            tau0=2.0*np.ones(len(t))

        tau = fsolve(dydt.taufunc,tau0, args=[zeta_x(t),zeta_y(t),Gamma(t)])

        return tau

    def thetaN(t):
        return dydt.thetaNfunc(zeta_x(t), tau(t))

    def thetaS(t):
        return dydt.thetaSfunc(zeta_y(t), tau(t))

    def gamma(t):
        return dydt.gammafunc(zeta_x(t), zeta_y(t), Gamma(t), thetaN(t))

    def g_x(t):
        return 0.5*(Gamma(t) + gamma(t))

    def g_y(t):
        return 0.5*(Gamma(t) - gamma(t))

    return zeta_x, zeta_y, Gamma, gamma, g_x, g_y, tau, thetaN, thetaS

#===============================================================================
def plot_stable_manifold(dydt, solall, Old_SS, New_SS, psi_choice, i, j):
    """
    Use the time paths of zetax(t), zetay(t), Gamma(t) to plot the
     "stable manifold" in a 3D space: the x-axis measures zetax;
                                      the y-axis measures zetay;
                                      the z-axis measures Gamma.
    Note: 1) zetax and zetay are predetermined state variable at any mmoment,
          while Gamma (defined as the sum of gx and gy) is a "jump variable."

    """
    zeta_x, zeta_y, Gamma, _, _, _, _, _, _  =\
    timepaths(dydt, solall, psi_choice, i, j)

    fig = plt.figure()

    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')

    # Retrieve data from time paths in transition
    t = np.linspace(0,100, 100) 
    x = zeta_x(t)
    y = zeta_y(t)
    z = Gamma(t)

    #Data at t=0
    zetax0 = zeta_x(0); zetay0 = zeta_y(0)
    Gamma0_jump = Gamma(0); Gamma0 = Old_SS[2][1][2]

    #Data at t-->inf (Pick t=100 as a approximation of infinity)
    zetax1 = zeta_x(100); zetay1 = zeta_y(100); Gamma1 = Gamma(100)

    
    print("\nChecks on steady states (Initial & New) and the jump in Gamma")
    print("---------------------------------------------------------------")
    print("zetax(0)        = ", zetax0)
    print("zetax(100)      = ", zetax1)
    
    print("zetay(0)        = ", zetay0)
    print("zetay(100)      = ", zetay1)

    print("Gamma(0)_jump   = ", Gamma0_jump)
    print("Gamma(100)      = ", Gamma1)
    print("Old_SS[2][1][2]  = ", Gamma0)
    print("---------------------------------------------------------------")
    print("")
    # Plot
    ax.plot3D (x, y, z, 'green', lw=2.5)
    ax.scatter([zetax0],[zetay0],[Gamma0_jump],c='r', marker='o', s=40)
    ax.scatter([zetax0], [zetay0], [Gamma0], s=40,facecolors='none',edgecolors='r')
    ax.scatter([zetax1], [zetay1], [Gamma1], c='r', marker='o', s=40)
    ax.scatter([0.80],[0.40],[0.15], c='k', marker='o', s=40)
    ax.set_xlabel(r'$\zeta_x$', fontsize=12)
    ax.set_ylabel(r'$\zeta_y$', fontsize=12)
    ax.set_zlabel(r'$\Gamma$', fontsize=12)
    ax.set_title('Stable Manifold as my=0.10--> mx=0.02, given psi=1.00')
    ax.set_xlim([0.80, 0.84])
    ax.set_ylim([0.80, 0.40])
    ax.set_zlim(0.15,0.19)
    ax.tick_params(axis='x', labelrotation=75)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.zaxis.set_tick_params(labelsize=7)
    plt.show()


def plot_timepaths(dydt, solall, Old_SS, New_SS, psi_choice, i, j):

    SteadyState_0 = Old_SS[i][j]      #Index i=1 refers to Experiment 2
    SteadyState_1 = New_SS[i][j]

    Gamma0, gx0, gy0, zetax0, zetay0, tau0 = SteadyState_0[[2,3,4,5,6,7]]
    Gamma1, gx1, gy1, zetax1, zetay1, tau1 = SteadyState_1[[2,3,4,5,6,7]]

    zeta_x, zeta_y, Gamma, gamma, g_x, g_y, tau, thetaN, thetaS \
    = timepaths (dydt, solall, psi_choice, i, j)

    t = np.linspace(0,50,200)

    
    fig0=plt.figure(0)

    plt.subplot(3,2,1)
    plt.scatter([0], [g_x(0)], s=80, facecolors='b', edgecolors='b')
    plt.scatter([0], [gx0], s=80, facecolors='none', edgecolors='b')
    plt.plot(t, g_x(t), 'b', lw=2.5 )
    plt.axhline(y=gx1, xmin=0, xmax=1, ls='dashed', color='b')
    plt.title("(a) Innovation rate of goods x")
    plt.xlim(-2, 40)
    plt.ylim(0.08,0.11)
    plt.text(6, 0.0935, r'$g_x(t)$', fontsize=16)

    plt.subplot(3,2,2)                                                  
    plt.scatter([0], [g_y(0)], s=80, facecolors='r', edgecolors='r')    
    plt.scatter([0], [gy0], s=80, facecolors='none', edgecolors='r')    
    plt.plot(t, g_y(t), 'r', lw=2.5 )                                   
    plt.axhline(y=gy1, xmin=0, xmax=1, ls='dashed', color='r')          
    plt.title("(b) Innovation rate of goods y")                         
    plt.xlim(-2, 40)                                                    
    plt.ylim(0.06,0.11)                                                 
    plt.text(2.5, 0.09, r'$g_y(t)$', fontsize=16)                       
                                                                                
    plt.subplot(3,2,3)                                                  
    plt.scatter([0], [Gamma0], s=80, facecolors='none', edgecolors='b') 
    plt.scatter([0], [Gamma(0)], s=80, facecolors='b', edgecolors='b')  
    plt.plot(t, Gamma(t), 'b', lw=2.5)                                  
    plt.axhline(y=Gamma1, xmin=0, xmax=1, ls='dashed', color='b')       
    plt.xlim(-2, 40)                                                    
    plt.ylim(0.15,0.20)                                                 
    plt.title("(c) Sum of innovation rates")                            
    plt.text(5, 0.177, r'$\Gamma(t)$', fontsize=16)                     

    plt.subplot(3,2,4)
    plt.scatter([0], [tau0], s=80, facecolors='none', edgecolors='b')
    plt.scatter([0], [tau(0)], s=80, facecolors='b', edgecolors='b')
    plt.plot(t, tau(t), 'g', lw=2.5)
    plt.axhline(y=gx1, xmin=0, xmax=1, ls='dashed', color='b')
    plt.axhline(y=tau1, xmin=0, xmax=1, ls='dashed', color='r')
    plt.title("(d) North-South terms of trade")
    plt.xlim(-2, 40)
    plt.ylim(2, 4.5)
    plt.text(3, 3.4, r'$\tau(t)$', fontsize=16)

    plt.subplot(3,2,5)
    plt.plot(t, zeta_x(t), 'b', lw=2.5 )
    plt.axhline(y=zetax1, xmin=0, xmax=1, ls='dashed', color='b')
    plt.xlim([-2, 40])
    plt.ylim([0.8, 0.9])
    plt.xlabel(r'$t$', fontsize=16)
    plt.title("(e) Fraction of monoply goods x")
    plt.xlabel(r'$t$', fontsize=16)
    plt.text(5, 0.837, r'$\zeta_x(t)$', fontsize=16)

    plt.subplot(3,2,6)
    plt.plot(t, zeta_y(t), 'r', lw=2.5 )
    plt.axhline(y=zetay1, xmin=0, xmax=1, ls='dashed', color='r')
    plt.xlim([-2, 40])
    plt.ylim([0.3, 0.9])
    plt.xlabel(r'$t$', fontsize=16)
    plt.title("(f) Fraction of monoply goods y")
    plt.xlabel(r'$t$', fontsize=16)
    plt.text(5, 0.5, r'$\zeta_y(t)$', fontsize=16)
    fig0.tight_layout()

    plt.show()

#===============================================================================

def intertemporal_welfare(dydt,solall,Old_SS,New_SS,psi_choice,i,j,my_choice):

    dydt.setpsi(psi_choice[i])  #i=2 refers to psi=1.00

    # Retrieve solution for an experiment corresponding to indexes i & j.
    # Note that the benchmark experiment corresponds to i=2 and j=1.
    # For the benchmark experiment, psi = psi_choice[i=2] = 1.00 and
    # j=1 refers to my0=0.10 being raised to my1=0.02, matching mx=0.01.

    # Retrive solution obtained by solving the dynamical system using Python's
    # BVP solver, scipy.integrate.solve_bvp. This solution is retrieved below:
    
    solution = solall[i]
    sol = solution[j].sol        

    # Retrieve steady-state solutions by solving the the dynamical system's
    # steady-state equilibrium using Python's solver, scipy,optimize.fsolve.
    # This solution is retrieved below:

    SteadyState_0 = Old_SS[i][j]  
    SteadyState_1 = New_SS[i][j]

    Gamma0, gx0, gy0, zetax0, zetay0, tau0 = SteadyState_0[[2,3,4,5,6,7]]
    Gamma1, gx1, gy1, zetax1, zetay1, tau1 = SteadyState_1[[2,3,4,5,6,7]]

    # Retrieve time paths of some endogenous variables that determine Omega
    zeta_x, zeta_y, Gamma, gamma, g_x, g_y, tau, thetaN, thetaS \
    = timepaths (dydt, solall, psi_choice, i, j)

    # Compute Omega in transition to steady state for North & South
    # Note: Omega is the measure of intertemporal welfare change in the time
    #       domain [0, inf). Choose tmax=500 to be an approximation of inf.
    #       the grater the chosen tmax, the more precise is the Omega measure.

    tmax = 500 

    SteadyState_0 = SteadyState_0[2:10]

    N_Omega_Transition = \
    dydt.Transition_Omega_N(Gamma, g_x, zeta_x, tau, tmax, SteadyState_0)
        
    S_Omega_Transition = \
    dydt.Transition_Omega_S(Gamma, g_y, zeta_y, tau, tmax, SteadyState_0)

    return N_Omega_Transition, S_Omega_Transition

#===============================================================================

def displaySS(SteadyState_Senarios):                                            
                                                                                
    SteadyState_Senarios = SteadyState_Senarios.T                               
                                                                                
    print ("\nTable 1: Four steady states with different pairs of (mx, my)")    
    print ("------------------------------------------------------------------")
    index = ['mx','my','Gamma','gx','gy','zetax','zetay','tau','thetax',\
    'thetay','Tx','Ty', 'wN/wS']                                                
    col = ['Senario 1','Senario 2','Senario 3','Senario 4']                     
    df = pd.DataFrame(SteadyState_Senarios, index=index, columns=col)           
    print(df)                                                                   
    print("")                                                                   

#===============================================================================                 

def displaySS_welfare(my_choice, N_Omega_ss_all, S_Omega_ss_all):               

    print("\nTable 2: North-South Steady-State Welfare Effects")                
    print("======================================================")             
    for i in range(3):                                                          
        data = np.array([N_Omega_ss_all[i], S_Omega_ss_all[i]])                 
        print("Experiment %1g: matching my=%4.2f to %4.2f=mx"\
        % (i+1, my_choice[i+1], my_choice[0]))                                  
        index=["Omega","Prod.Innovation","Terms of Trade","Mkt Power","Saving"] 
        print(pd.DataFrame(data.T, index=index, columns = ["North","South"]))   
        if i<=1: print("")                                                      
    print("======================================================")             
    print("")         

#===============================================================================

def displayOmega_Transition (N_Omega, S_Omega, psi_choice, i, j, my_choice):
    """
    OmegaTrans_N, ProdAvailTrans_N, TermsOfTradeTrans_N,
    MktPowerTrans_N, RandDTrans_N = N_Omega_Transition 
    
    OmegaTrans_S, ProdAvailTrans_S, TermsOfTradeTrans_S,
    MktPowerTrans_S, RandDTrans_S = S_Omega_Transition
    """
    data = np.array([N_Omega, S_Omega])
    
    index=["Omega","Product Innovation","Terms of Trade","Mkt Power","Saving"]
    
    psi = psi_choice[i]

    my0 = my_choice[j+1]

    my1 = my_choice[0]

    if j==0:
        print("\nTable 3: North-South Intertemporal-Welfare Effects")
        print("==========================================================")
    print("Experiment %1g: my=%4.2f --> %4.2f, given psi=%3.2f" \
          % (j+1, my0, my1, psi)) 
    print(pd.DataFrame(data.T, index=index, columns = ["North","South"]))
    if j==2:
        print("==========================================================")
    print("")

#===============================================================================

def robustness_checks(dydt,solall,Old_SS,New_SS,psi_choice,i,j,my_choice):
    """
    This function is to plot how each region's intertemporal welfare
    change measured by Omega in transition to steady state is sensitive to
    the coefficient of creative destruction, psi, in each of the three
    experiments:

    Experiment 1: my=0.05 --> 0.02
    Experiment 2: my=0.10 --> 0.02
    Experiment 3: my=0.2i0 --> 0.02

    In the main(), the three experiments use the benchmark value of psi=1.00.

    By changing the value of psi in the main(),  one can obtain the
    intertempoal welfare change for the North and South, respectively.
    """
    Omega_N1 = []; Omega_N2=[]; Omega_N3=[]
    Omega_S1 = []; Omega_S2=[]; Omega_S3=[]

    for i in range(5):

        for j in range(3):
            North, South \
            =intertemporal_welfare(dydt,solall,Old_SS,New_SS,psi_choice,i,j,\
                                    my_choice)

            if j==0:
                Omega_N1.append(North[0])
                Omega_S1.append(South[0])
            if j==1:
                Omega_N2.append(North[0])
                Omega_S2.append(South[0])
            if j==2:
                Omega_N3.append(North[0])
                Omega_S3.append(South[0])

    plt.figure(1)
    plt.plot(psi_choice, Omega_S1, 's--', lw=2, ms=9)
    plt.plot(psi_choice, Omega_S2, 'o-', lw=2, ms=10)
    plt.plot(psi_choice, Omega_S3, '^--', lw=2, ms=10)
    plt.legend([r'$m_y:\,0.20\rightarrow 0.02$',\
    r'$m_y:\,0.10\rightarrow 0.02$',\
    r'$m_y:\,0.05\rightarrow 0.02$'],fontsize=16)
    plt.axhline(0, ls='dashed')
    plt.xlim([0.4,1.4])
    plt.xlabel(r'$\mathrm{creative\, destruction}, \,\psi$', fontsize=22)
    plt.ylabel(r'$\Omega^S$', fontsize=22)

    plt.figure(2)
    plt.plot(psi_choice, Omega_N1, 's--', lw=2, ms=9)
    plt.plot(psi_choice, Omega_N2, 'o-', lw=2, ms=10)
    plt.plot(psi_choice, Omega_N3, '^--', lw=2, ms=10)
    plt.legend([r'$m_y:\,0.20\rightarrow 0.02$',\
    r'$m_y:\,0.10\rightarrow 0.02$',\
    r'$m_y:\,0.05\rightarrow 0.02$'],fontsize=16, loc='lower right')
    plt.axhline(0,ls='dashed')
    plt.xlim([0.4,1.4])
    plt.ylim(-0.8, 0)
    plt.xlabel(r'$\mathrm{creative\, destruction},\,\psi$', fontsize=22)
    plt.ylabel(r'$\Omega^N$', fontsize=22)

    plt.show()

#===============================================================================

main()    # Call the driver to run the entire Python program.
   
