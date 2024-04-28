####
# Trajectories for the SIR model in Exercise 3
####

import numpy as np
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

t_final =  1

# <------------------------------>

# <------- fill in the reaction rate functions ---->
#reaction propensities
def propensities(X,k):
        R = np.zeros((3,1))
        R[0] = k[0]*X[0]*X[1]
        R[1] = k[1]
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

def Find_Reaction_Index(a):
	"""	
	@brief The function takes in the reaction rate vector and returns
	the index of the reaction to be fired of a possible reaction candidate.
	@param a : Array (num_reaction,1) 

	"""
	# small hack as the numpy uniform random number includes 0
	r = np.random.rand()
	while r == 0:
		r = np.random.rand()

	return np.sum(np.cumsum(a) < r*np.sum(a))

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

        #for storage
	X_store = []
	T_store = []
	#initialize
	t = 0.0
	x = X0
	X_store.append(x[1,0])
	T_store.append(t)

	while t < t_final:
                 #compte reaction rate functions
		a = propensities(x,k)
		# 1. When? Compute first Jump Time
		tau = Time_To_Next_Reaction(np.sum(a))
		
		""" Stopping criterium: Test if we have jumped too far and if
		yes, return the stored variables (states, times)
		"""
		if (t + tau > t_final) or (np.sum(a) == 0):
			return np.array(X_store),np.array(T_store)
		else:
			# Since we have not, we need to find the next reaction
			t = t + tau #update time
			#2. What? find reaction to execute and execute the reaction
			j = Find_Reaction_Index(a)
			x = x + Stochiometry[:,[j]]
			# Update Our Storage
			X_store.append(x[1,0])
			T_store.append(t)

# Run a number of simulations and save the respective trajectories
for i in range(NrSimulations):
	# get a single realisation
	states, times = SSA(S,X0,t_final,k)
	##a) save trajectory
	Output = np.concatenate((np.array(times,ndmin=2),np.array(states,ndmin=2)), axis=0)
	np.savetxt('Task2Traj'+str(i+1)+'.txt',Output,delimiter = ',',fmt='%1.3f');	
