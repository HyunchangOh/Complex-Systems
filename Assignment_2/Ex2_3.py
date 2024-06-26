####
# Trajectories for the SIR model in Exercise 3
####

import numpy as np
from math import sin, radians
#import matplotlib.pyplot as pl

#import the input file
X0 = np.loadtxt('Input.txt',ndmin=2) #initial state
In = np.loadtxt('Input2.txt') #seed and number of simulations
#set the seed and the number of simulations from the input file
np.random.seed(seed=int(In[0]))
NrSimulations = int(In[1])

#-----Fixed Quantities-----
# Stoichiometric matrix

# <------- fill in the model specifics ---->

#               r1   r2  r3 
S = np.array([  [-1,  0,  0],#X1
                [-1,  1,  1],#X2
                [0,   0, -1]])#X3

#reaction parameters
k = [0.01, 0.1, 0.01]

t_final =  10

# <------------------------------>

# <------- fill in the reaction rate functions ---->
#reaction propensities
def propensities(X,k,t):
        R = np.zeros((3,1))
        R[0] = k[0]*X[0]*X[1]
        #R[1] = k[1]*0.5*(sin(radians(180.0*t))+2.0)
        R[1] = k[1]*0.5*(sin(180*t)+2)
        R[2] = k[2]*X[1]*X[2]
        return R

def propensities_B(X,k):
        R = np.zeros((3,1))
        R[0] = k[0]*X[0]*X[1]
        R[1] = 0.15
        R[2] = k[2]*X[1]*X[2]
        return R
# <------------------------------>

def Time_To_Next_Reaction(lam):
	"""
	@brief The function samples from an exponential distribution with rate "lam". 
	@param lam : real value positive.
	"""

	# small hack as the numpy uniform random number includes 0
	r = np.random.rand()
	while r == 0:
		r = np.random.rand()

	return (1.0/lam)*np.log(1.0/r)

def Find_Reaction_Index(a,u2):
	"""	
	@brief The function takes in the reaction rate vector and returns
	the index of the reaction to be fired of a possible reaction candidate.
	@param a : Array (num_reaction,1) 

	"""
	# small hack as the numpy uniform random number includes 0

	return np.sum(np.cumsum(a) < u2*np.sum(a))

def SSA(Stochiometry,X0,t_final,k):
	"""
	@brief  The Stochastic Simulation Algorithm. Given the stochiometry,
	propensities and the initial state; the algorithm
	gives a stochastic trajectory of the Kurtz process until $t_final.$
	
	@param Stochiometry : Numpy Array (Num_species,Num_reaction).
	@param X_0: Numpy Array (Num_species, 1).
	@param t_final : positive number.
	@param k1,k2,k3,k4: positive numbers  (reaction rate parameters)

	"""

	X_store = []
	T_store = []
	#initialize
	t = 0.0
	x = X0
	X_store.append(x[1,0])
	T_store.append(t)

	while t < t_final:
		B = propensities_B(x,k)
		tau= Time_To_Next_Reaction(np.sum(B))
		if (t + tau > t_final) or (np.sum(B) == 0):
			return np.array(X_store),np.array(T_store)
		else:
			t = t + tau 
			a = propensities(x,k,t-tau)
			u2 = np.random.rand()
			while u2==0:
				u2= np.random.rand()
			if u2<np.sum(a)/np.sum(B):
				j = Find_Reaction_Index(a,u2)
				x = x + Stochiometry[:,[j]]
				X_store.append(x[1,0])
				T_store.append(t)

# Run a number of simulations and save the respective trajectories
for i in range(NrSimulations):
	# get a single realisation
	states, times = SSA(S,X0,t_final,k)
	##a) save trajectory
	Output = np.concatenate((np.array(times,ndmin=2),np.array(states,ndmin=2)), axis=0)
	np.savetxt('Task3Traj'+str(i+1)+'.txt',Output,delimiter = ',',fmt='%1.3f');	
