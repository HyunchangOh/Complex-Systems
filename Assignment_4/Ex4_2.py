####
# Trajectories for the SIR model in Exercise 3
####

import numpy as np
import scipy.integrate
#import matplotlib.pyplot as pl

NrSimulations = 1

X0 = np.array([200,0])

#-----Fixed Quantities-----
# Stoichiometric matrix

# <------- fill in the model specifics ---->

#               r1   r2  r3 
S = np.array([  [-1,  0],#X1
                [1,  -1]])#X3

#reaction parameters
k = [0.5, 0.3]

t_final =  24

# <------------------------------>

# <------- fill in the reaction rate functions ---->
#reaction propensities
def propensities(X,k):
        R = np.zeros((2,1))
        R[0] = k[0]*X[0]
        R[1] = k[1]*X[1]
        return R
# <------------------------------>

def RHS(t,X,k):
	R = propensities(X,k)
	dx = np.dot(S,R)
	return dx

timeout = []
t=0
for i in range(241):
	timeout.append(t)
	t+=0.1



sol = scipy.integrate.solve_ivp(RHS,[0,24.1],y0=X0,args=[k],vectorized=True,t_eval=timeout)



# Run a number of simulations and save the respective trajectories
for i in range(NrSimulations):
	# get a single realisation
	times = timeout
	states = sol['y'][1]
	##a) save trajectory
	Output = np.concatenate((np.array(times,ndmin=2),np.array(states,ndmin=2)), axis=0)
	np.savetxt('Task2Traj'+str(i+1)+'.txt',Output,delimiter = ',',fmt='%1.2f');	
