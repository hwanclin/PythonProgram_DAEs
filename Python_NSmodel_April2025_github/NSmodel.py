# Source File: NSmodel.py  
   
import numpy as np
from math import pow, log, exp
from scipy.optimize import fsolve
from scipy.integrate import quad

#===============================================================================
# This source file presents a Python class that mainly defines a North-South
# dynamical system and a series of class functions designed to compute the
# dynamical system's long-run steady state, transition dynamics, and
# intertemporal welfare for the North and South. The dynamical system features
# a system of differential algebraic equations (DAEs) as laid out below:
#
#   dzetax(t)
#  ---------- = gx * [1-zetax(t)] - mx * zetax(t)                          (1)
#      dt
#
#   dzetay(t)
#  ---------- = gy * [1-zetay(t)] - my * zetay(t)                          (2)
#      dt
#
#   dGamma(t)       LN
#  ---------- = [--------- - Gamma(t)] * 
#      dt        a*(1+psi)
#
#                            thetaN(t)*(eta-1)    LN
#   [rho+mx+(1+psi)*gx(t) - ----------------- * (--- - (1+psi)*Gamma(t))]  (3)
#                                zetax(t)          a
#
#  s.t.:
#
#                                     LN
#   (1+psi)*[gx(t)-gy(t)] - (eta-1)*[---- - (1+psi)*Gamma(t)] *
#                                     a
#       thetaN(t)     1-thetaN(t)
#     [----------- - ------------] + (mx - my) = 0                         (4)
#       zetax(t)       zetay(t) 
#
#                     zetax(t)*tau(t)^(1-epsilon)
#   thetaN(t) - ----------------------------------------- = 0              (5)
#                1-zetax(t) + zeta(t)*tau(t)^(1-epsilon)
#
#                          1-zetay(t)
#   thetaS(t) - ----------------------------------------- = 0              (6)
#                1-zetay(t) + zetay(t)*tau(t)^(1-epsilon)
#
#              1-thetaS(t)                 LS
#   tau(t) - [-------------] * [-----------------------]  = 0              (7)
#              1-thetax(t)       LN-a*(1+psi)*Gamma(t)
#
#   gx(t) + gy(t) - Gamma(t) = 0                                           (8)
#  
# where t is in [0, inf). We can solve the system of DAEs (1)-(8) by solving 
# a system of ordinary differential equations (1)-(3) in 
# [zeta(t), zetay(t), Gamma(t), subject to constraints (4)-(8).
#
# In total, the reduced-form North-South dynamical system is represented by
# 8 equations with 8 endogenous variables: zetax, zetay, Gamma, gx, gy, tau,
# thetaN, and thetaS, The definition of each variable is given below in remarks
# along with the code.

# In the constrained ODE system, Gamma(t) is a "jump" variable at any point in 
# time, but both zetax(t) and zetay(t) are pre-determined state variables. These
# state variables cannot make a discrete change at any moment. They can only 
# evolve continuously over time, in contrast to Gamma(t).
#
# Therefore, as a policy shock occurs at t=0, the constrained ODE system
# presents a boundary-value problem, subject to the system's two-point boundary
# conditions. To use the Python solver, scipy.integrate,solve_bvp, we need the 
# the following two-point boundary conditions: 
#
#   At t=0, zetax(t) = zetax0, an initial steady-state equilibrium value.
#           zetay(t) = zetay0  an initial steady-state equilibrium value,
#
#   At t--> inf, Gamma(t) converges to Gamma(inf). 
#
# For details about the dynamical system, please refer to the paper,
#
# "A dynamic modelling approach to North-South disparities in IPR protection"
# (by Hwan C. Lin), published in Oxford Economic Papers: 
#    
#                https://doi.org/10.1093/oep/gpaf024
#
# where eqs.(22)-(29) are exactly the same as eqs.(1)-(8) presented above.
#
# Python Coder: Professor Hwan C. Lin                     Date: April 15, 2025 
#===============================================================================

class NSmodel:

    def __init__(self, LN, LS, a, epsilon, rho, mx, my, eta, psi):
        """
        Nine parameters for the North-South model.  
        """
        self.LN      = LN           #Labor force in North
        self.LS      = LS           #Labor force in South
        self.a       = a            #Research productivity parameter
        self.epsilon = epsilon      #Elasticity of demand for a product
        self.rho     = rho          #Time preference coefficient
        self.mx      = mx           #Imitation rate of good x
        self.my      = my           #Imitation rate of good y
        self.eta     = eta          #Markup on unimitated goods
        self.psi     = psi          #Creative destruction coefficient

        self.ssy0 = []       #A variable list, ssy0, in initial steady state     
        self.ssy1 = []       #A variable list, ssy1, in new steady state

        # where ssy0 is to hold [zetax0, zetay0, Gamma0]
        #       ssy1 is to hold [zetax1, zetay1, Gamma1]


    def setmy(self, new_my): self.my = new_my

    def setmx(self, new_mx): self.mx = new_mx

    def setpsi(self, new_psi): self.psi = new_psi

    def setssy0(self, ssy0): self.ssy0 = ssy0

    def setssy1(self, ssy1): self.ssy1 = ssy1

    def steady_state(self, x):
        """
        This function defines a seven dimensional steady-state system based on 
        the North-South dynamical system.
        Input x - an array of 7 unknowns.
        Output res - an array of 7 residuals.
        """

        #==========================================
        # 8 parameters with eta=epsilon/(epsilon-1)
        #==========================================
        LN, LS = self.LN, self.LS
        a, epsilon, rho = self.a, self.epsilon, self.rho
        mx, my, eta, psi  = self.mx, self.my, self.eta, self.psi

        #=============================================================
        # 7 endogenous variables in long-run steady-state equilibrium
        #=============================================================
        gx     = x[0]   #Innovation rate of x-type goods
        gy     = x[1]   #Innovation rate of y-type goods
        zetax  = x[2]   #Unimitated fraction of x-type goods
        zetay  = x[3]   #Unimitated fraction of y-type goods
        tau    = x[4]   #North-South terms of trade
        thetaN = x[5]   #Share of North's manufacturing labor for x-type goods        
        thetaS = x[6]   #Share of South's manufacturing labor for y-type goods
        
        #======================
        # 2 Auxiliary variables
        #======================
        Gamma = gx + gy
        gamma = gx - gy

        #====================================================
        # 7 Long-run steady-state equilibrium conditions
        #====================================================
        res = np.zeros(7)
        
        #Use eq.(3) to set dGamma/dt = 0, yielding res[0]:
        res[0] = rho + mx + (1 + psi)*0.5*(Gamma + gamma) \
                 - thetaN*(eta - 1)*(LN/a - (1+psi)*Gamma)/zetax
        
        #Use eq.(4) that governs gamma = gx - gy to get res[1]:
        res[1] = (1+psi)*gamma + (mx-my) - \
           (eta-1)*(LN/a - (1+psi)*Gamma)*(thetaN/zetax - (1-thetaN)/zetay)
        
        #Use eq.(1) to set dzetax/dt = 0, yielding res[2]:
        res[2] = 0.5*(Gamma+gamma)*(1.0-zetax) - mx*zetax

        #Use eq.(2) to Set dzetay/dt = 0, yielding res[3]:
        res[3] = 0.5*(Gamma-gamma)*(1.0-zetay) - my*zetay
        
        #Use eq.(7) that governs tau to get res[4]:
        res[4] = tau - ((1.0-thetaS)/(1.0-thetaN)) * (LS/(LN-a*(1+psi)*Gamma))
        
        #Use eq.(5) that governs thetaN to get res[5]:
        coef = 1.0-zetax+zetax*tau**(1.0-epsilon)
        res[5] = thetaN - zetax*tau**(1.0-epsilon) / coef

        #Use eq.(6) that governs thetaS to get res[6]:
        res[6] = thetaS - (1.0-zetay) / (1.0-zetay+zetay*tau**(1.0-epsilon))

        return res

        #----------------------------------------------------------------------- 
        # Note: The steady-state equilibrium conditions do not include gx and gy
        #       explicitly, This is because we the introduce the auxiliary 
        #       variable gamma, which is defined as gx - gy. 
        #       So, with Gamma = gx + gy, we can use Gamma and gamma to
        #       determine gx and gy according to::
        #
        #          gx = 0.5*(Gamma + gamma) and gy = 0.5*(Gamma - gamma)
        #-----------------------------------------------------------------------

    def __call__(self, t, y):
        """
        This function defines a constrained system of Ordinary Differential 
        Equations (ODEs) based on eqs.(1)-(8) mentioned above. The function
        __call__(...) is a callable function object. Hence, a class object
        initiated elsewhere (say, in the driver file) can represent the 
        constrained system of ODEs.

        Input: t is a time variable; y is an array of 3 elements: 
                y[0] = zetax
                y[1] = zetay
                y[2] = Gamma
        Output: an array of size 3.
        """
        LN, LS = self.LN, self.LS
        a, epsilon, rho = self.a, self.epsilon, self.rho
        mx, my, eta, psi  = self.mx, self.my, self.eta, self.psi

        #-----------Three unknowns---------------------------------------------
        zetax = y[0]; zetay = y[1]; Gamma = y[2]
        

        #-----------Static equilibrium in transition --------------------------
        #tau = np.zeros(t.size)
        #for i in range(t.size):
        #    tau[i] = fsolve(self.taufunc, 1.2, args=y)[0]
        
        if np.size(t)==1:
            tau0 = 2.0
        else:
            tau0=2.0*np.ones(len(t))
        tau = fsolve(self.taufunc,tau0, args=y)
                
        thetaN = self.thetaNfunc(zetax, tau)
        thetaS = self.thetaSfunc(zetay, tau)
        gamma = self.gammafunc(zetax, zetay, Gamma, thetaN)
        #---------------------------------------------------------------------- 
        

        #----------------------------------------------------------------------
        # The North-South dynamical system: 3 first-order ODEs
        #----------------------------------------------------------------------
        # auxiliary variables:
        coef1 = LN/a - (1+psi)*Gamma
        coef2 = thetaN * (eta - 1)
        coef3 = 1.0/(1+psi)
        gx = 0.5 * (Gamma + gamma)
        gy = 0.5 * (Gamma - gamma)

        # 3 ODEs determine the dynamics of zetax(t), zetay(t), and Gamma(t) 
        dydt0 = gx * (1-zetax) - mx * zetax
                        
        dydt1 = gy * (1-zetay) - my * zetay
                   
        dydt2 = coef3 * coef1 * (rho + mx + (1+psi)*gx - coef2 * coef1 / zetax)
        #----------------------------------------------------------------------
        
        return np.array([dydt0, dydt1, dydt2])

    
    def bc (self, ya, yb):
        """
        Two-point boundary conditions at t=0 and t->inf
        """
        zetax0, zetay0 = self.ssy0[0], self.ssy0[1]
        Gamma1 = self.ssy1[2]

        res = np.zeros(3)

        res[0] = ya[0] - zetax0 #left condition for zetax at t=0
        res[1] = ya[1] - zetay0 #left condition for zetay at t=0
        res[2] = yb[2] - Gamma1 #right condition for Gamma at t->inf

        return  res

    
    #--------------------------------------------------------------------------
    # Static equilibrium conditions in transition
    #--------------------------------------------------------------------------
    def taufunc(self, x, y):
        """
        This residual function, taufunc(...), is obtained by substituting 
        eqs.(5)-(6) into eq.(7). 

        Input: x - the terms of trade to be solved
               y - a vector of [zetax, zetay, Gamma]
        Output: res - a residual scalar 

        """
        LN, LS, a = self.LN, self.LS, self.a
        epsilon, psi = self.epsilon, self.psi
        
        tau = x 
        zetax, zetay, Gamma = y
        
        coef1 = 1.0 - zetax + zetax * tau**(1.0-epsilon)
        coef2 = 1.0 - zetay + zetay * tau**(1.0-epsilon)
        labor_ratio = LS/(LN - a*(1+psi)*Gamma)
        
        res = tau**epsilon - (zetay/(1.0-zetax)) * (coef1/coef2) * (labor_ratio)

        return res

    def thetaNfunc(self, zetax, tau):
        """
        This function determines the equilibrium value of thetaN at any moment
        according to eq.(5)
        """
        epsilon = self.epsilon
        coef = 1.0 - zetax + zetax * tau**(1.0-epsilon)

        return zetax * tau**(1.0-epsilon) / coef
        
    def thetaSfunc(self, zetay, tau):
        """
        This function determines the equilibrium value of thetaS at any moment
        according to eq.(6).
        """
        epsilon = self.epsilon
        return (1.0 - zetay) / (1.0 - zetay + zetay * tau**(1.0-epsilon))

    def gammafunc(self, zetax, zetay, Gamma, thetaN):
        """
        This function determines the equilibrium value of gamma = gx-gy at any
        moment according to eq,(4). 
        """

        eta, a, LN, mx, my = self.eta, self.a, self.LN, self.mx, self.my
        psi = self.psi 

        coef = thetaN/zetax - (1-thetaN)/zetay
        diff = (eta-1) * (LN/a - (1+psi)*Gamma) * coef - (mx-my)

        gamma = diff/(1.0+psi)
        
        return gamma
    #--------------------------------------------------------------------------
    

    #--------------------------------------------------------------------------
    # Functions for the computations of steady-state welfare changes
    #--------------------------------------------------------------------------    
    def Steady_Omega(self, steadystate_0, steadystate_1):
        """
        This function computes the changes in steady-state welfare and its
        welfare components according to eqs.(31) and (32a)-(32d) for the North
        and eqs.(33) and (34a)-(34d) for the South in the paper. For these
        computations, transition dynamics are disregarded.

        Input: steadystate_0 - the initial steady state, a list of size 8.
               steadystate_1 - the new steady state, a list of size 8.
        Output: Steady_Omega_N - a list of size 5 for the North.
                Steady_Omega_S - a list of size 5 for the South.
        """
        epsilon, a, psi = self.epsilon, self.a, self.psi
        Gamma0, gx0, gy0, zetax0, zetay0, tau0, thetaN0, thetaS0 = steadystate_0
        Gamma1, gx1, gy1, zetax1, zetay1, tau1, thetaN1, thetaS1 = steadystate_1
        
        #----------------------------------------------------------------------
        # Changes in steady-state welfare and its components for the North
        #----------------------------------------------------------------------
        # Innovation Effect:
        ProdAvail_N = exp( (1.0/(epsilon-1.0)) * (gx1 - gx0) ) #rho is out
        
        # Saving Effect:
        RandD_N = exp( log((1.0 - a*(1+psi)*Gamma1)/(1.0 - a*(1+psi)*Gamma0))) 
        
        # Terms-of-Trade Effect:
        coef1 = (1.0 - zetax0) * tau1**(epsilon - 1.0) + zetax0
        coef2 = (1.0 - zetax0) * tau0**(epsilon - 1.0) + zetax0
        TermsOfTrade_N = exp( (1.0/(epsilon-1.0)) * log(coef1/coef2) )
        
        # Monoply or Market-Power Effect:
        coef3 = (1.0 - zetax1) * tau1**(epsilon - 1.0) + zetax1
        coef4 = (1.0 - zetax0) * tau1**(epsilon - 1.0) + zetax0
        MktPower_N = exp( (1.0/(epsilon - 1.0)) * log(coef3/coef4) )

        # The North's Steady-state welfare change, measured by OmegaSS_N
        OmegaSS_N = ProdAvail_N * RandD_N * TermsOfTrade_N * MktPower_N - 1

        # Steady-state welfare solution set for the North
        Steady_Omega_N = [ OmegaSS_N,
                           ProdAvail_N,
                           TermsOfTrade_N,
                           MktPower_N,
                           RandD_N  ]
        
        #----------------------------------------------------------------------
        # Changes in steady-state welfare and its components for the South
        #----------------------------------------------------------------------
        # Innovation Effect
        ProdAvail_S = exp( (1.0/(epsilon-1.0)) * (gy1 - gy0) ) # rho is out.
        
        # Terms-of-Trade Effect
        coef1 = 1.0 - zetay0 + zetay0 * tau1**(1.0-epsilon)
        coef2 = 1.0 - zetay0 + zetay0 * tau0**(1.0-epsilon)
        TermsOfTrade_S = exp( (1.0/(epsilon-1.0)) * log(coef1/coef2) )
        
        # Monopoly or Market-Power Effect
        coef3 = 1.0 - zetay1 + zetay1 * tau1**(1.0-epsilon)
        coef4 = 1.0 - zetay0 + zetay0 * tau1**(1.0-epsilon)
        MktPower_S = exp( (1.0/(epsilon-1.0)) * log(coef3/coef4) )
        
        # The South's Steady-state welfare change, measured by OmegaSS_S
        OmegaSS_S = ProdAvail_S * TermsOfTrade_S * MktPower_S - 1
        
        # Steady-state welfare solution set for the South
        Steady_Omega_S = [ OmegaSS_S,
                           ProdAvail_S,
                           TermsOfTrade_S,
                           MktPower_S,
                           None  ]

        return Steady_Omega_N, Steady_Omega_S


    #--------------------------------------------------------------------------
    # Functions for the computations of intertemporal welfare changes
    #--------------------------------------------------------------------------
    def Transition_Omega_N(self, Gamma, gx, zetax, tau, tmax, steadystate_0):
        """
        This function computes the changes in Northern intertemporal welfare 
        and its welfare components according to eqs.(31) and (32a)-(32d) in the
        paper. For these computations, transition dynamics are taken into
        account.

        Input: Gamma = Gamma(t), where t in [0, inf)
               gx    = gx(t)
               zetax = zetax(t)
               tau   = tau(t)
               tamx  = the chosen approximation of inf
               steadystate_0 - the initial steady state, a list of size 8.
        Output: Transition_Omega_N - a list of size 5 for the North.
        """
        rho, a, epsilon, psi = self.rho, self.a, self.epsilon, self.psi
        
        # Retrieve intial steady-state values
        Gamma0, gx0, gy0, zetax0, zetay0, tau0, thetaN0, thetaS0 = steadystate_0


        # Innovation Effect from t=0 to t=tmax:
        def f_ProdAvail(t):
            return np.exp(-rho*t) * (quad(gx,0,t)[0] - t*gx0)   
        
        ProdAvail_N = exp((rho/(epsilon-1.0))*quad(f_ProdAvail,0,tmax)[0])


        # Terms-of-Trade Effect from t=0 to t=tmax
        def f_TermsOfTrade(t):
            coef1 = (1.0 - zetax0) * tau(t)**(epsilon - 1.0) + zetax0
            coef2 = (1.0 - zetax0) * tau0**(epsilon - 1.0) + zetax0
            return exp(-rho*t) * log(coef1/coef2)
        
        TermsOfTrade_N = exp((rho/(epsilon-1.0))*quad(f_TermsOfTrade,0,tmax)[0])

        
        # Monopoly or Market-Power Effect from t=0 to t=tmax 
        def f_MktPower(t):
            coef3 = (1.0 - zetax(t)) * tau(t)**(epsilon - 1.0) + zetax(t)
            coef4 = (1.0 - zetax0) * tau(t)**(epsilon - 1.0) + zetax0
            return exp(-rho*t) * log(coef3/coef4)
        
        MktPower_N = exp( (rho/(epsilon-1.0)) * quad(f_MktPower, 0, tmax)[0] )

        # Saving Effect from t=0 to t=tmax
        def f_RandD(t):
            coef = log((1.0-a*(1+psi)*Gamma(t))/(1.0-a*(1+psi)*Gamma0)) 
            return exp(-rho*t) * coef 

        RandD_N = exp( rho * quad(f_RandD, 0, tmax)[0] )

        # The North's intertemporal welfare change, measured by Omega_N
        Omega_N =  ProdAvail_N * TermsOfTrade_N * MktPower_N * RandD_N - 1

        # Intertemporal welfare solution set for the North
        Transition_Omega_N = [ Omega_N,
                               ProdAvail_N,
                               TermsOfTrade_N,
                               MktPower_N, 
                               RandD_N  ]

        return Transition_Omega_N

    def Transition_Omega_S(self, Gamma, gy, zetay, tau, tmax, steadystate_0):
        """
        This function computes the changes in Southern intertemporal welfare 
        and its welfare components according to eqs.(33) and (34a)-(34d) in the
        paper. For these computations, transition dynamics are taken into
        account.

        Input: Gamma = Gamma(t), where t in [0, inf)
               gx    = gx(t)
               zetax = zetax(t)
               tau   = tau(t)
               tamx  = the chosen approximation of inf
               steadystate_0 - the initial steady state, a list of size 8.
        Output: Transition_Omega_S - a list of size 5 for the South.
        """
        rho, a, epsilon, psi = self.rho, self.a, self.epsilon, self.psi
        
        Gamma0, gx0, gy0, zetax0, zetay0, tau0, thetaN0, thetaS0 = steadystate_0

        # Innovation Effect from t=0 to t=tmax    
        def f_ProdAvail(t):
            return exp(-rho*t) * (quad(gy,0,t)[0] - t*gy0)   
        
        ProdAvail_S = exp((rho/(epsilon-1.0)) * quad(f_ProdAvail, 0, tmax)[0])

        # Terms-of-Trade Effect from t=0 to t=tmax
        def f_TermsOfTrade(t):
            coef1 = 1.0 - zetay0 + zetay0 * tau(t)**(1.0 - epsilon)
            coef2 = 1.0 - zetay0 + zetay0 * tau0**(1.0 - epsilon)
            return exp(-rho*t) * log(coef1/coef2)
        
        TermsOfTrade_S = exp((rho/(epsilon-1.0))*quad(f_TermsOfTrade,0,tmax)[0])

        # Monopoly or Market_Power Effect from t=0 to t=tmax
        def f_MktPower(t):
            coef3 = 1.0 - zetay(t) + zetay(t) * tau(t)**(1.0 - epsilon)
            coef4 = 1.0 - zetay0 + zetay0 * tau(t)**(1.0 - epsilon)
            return exp(-rho*t) * log(coef3/coef4)
        
        MktPower_S = exp( (rho/(epsilon-1.0)) * quad(f_MktPower, 0, tmax)[0] )

        # The South's intertemporal welfare change, measured by Omega_S  
        Omega_S =  ProdAvail_S * TermsOfTrade_S * MktPower_S - 1

        # Intertemporal welfare solution set for the South
        Transition_Omega_S = [ Omega_S, 
                               ProdAvail_S, 
                               TermsOfTrade_S, 
                               MktPower_S, 
                               None  ]

        return Transition_Omega_S 
    #--------------------------------------------------------------------------    

    #          ***** The End of the Class (NSModel) *****

#===============================================================================




