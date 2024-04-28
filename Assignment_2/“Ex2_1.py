# provided to you by the best bioinformatics lecturer you'll ever have (starts with M, ends with t)

import numpy as np

# A: -----Fixed Quantities-----
#0. initial state
X0 = np.loadtxt('Input.txt')

# ===> fill here, everywhere where a "..." is <===

#1. Stoichiometric matrix
S = np.array([
[-1,0,0],
[-1,1,1],
[0,0,-1]
]) # !!check dimension of the array!!

#2. reaction parameters
k = [0.01, 0.1, 0.01]


# B: functions that depend on the state of the system X
def ReactionRates(k,X):
        R = np.zeros((3,1))
        R[0] = k[0]*X[0]*X[1]
        R[1] = k[1]
        R[2] = k[2]*X[1]*X[2]
        return R
# ===>       -----------------------     <===

# compute reaction propensities/rates
R = ReactionRates(k,X0)

#compute the value of the ode with time step delta_t = 1
dX = np.dot(S,R)

##a) save stoichiometric Matrix
np.savetxt('SMatrix.txt',S,delimiter = ',',fmt='%1.0f');
##b) save ODE value as float with 2 digits after the comma (determined with the c-style precision argument e.g. '%3.2f')
np.savetxt('ODEValue.txt',dX,delimiter=',',fmt='%1.2f');

