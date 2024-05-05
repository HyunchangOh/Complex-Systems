####
# Trajectories for the SIR model in Exercise 3
####

import numpy as np
#import matplotlib.pyplot as pl

#import the input file
X0 = np.array([
	[10],
	[10]
])
In = np.loadtxt('Input.txt') #seed and number of simulations
#set the seed and the number of simulations from the input file
np.random.seed(seed=int(In[0]))
NrSimulations = 300

#-----Fixed Quantities-----
# Stoichiometric matrix

# <------- fill in the model specifics ---->

#               r1   r2  r3 
S = np.array([  [1,  0,  -1],#X1
                [0,  1,  -1],#X2
])#X3

#reaction parameters
k = [1,1, 0.01]

t_final =  100

# <------------------------------>

# <------- fill in the reaction rate functions ---->
#reaction propensities
def propensities(X,k):
        R = np.zeros((3,1))
        R[0] = k[0]
        R[1] = k[1]
        R[2] = k[2]*X[0]*X[1]
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
	X2_store = []
	T_store = []
	#initialize
	t = 0.0
	x = X0
	X_store.append(x[0,0])
	X2_store.append(x[1,0])
	# print(x[0,0])
	# print(x[1,0])
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
			return np.array(X_store),np.array(X2_store),np.array(T_store)
		else:
			# Since we have not, we need to find the next reaction
			t = t + tau #update time
			#2. What? find reaction to execute and execute the reaction
			j = Find_Reaction_Index(a)
			x = x + Stochiometry[:,[j]]
			# Update Our Storage
			X_store.append(x[0,0])
			X2_store.append(x[1,0])
			T_store.append(t)

X0_stateses = []
# Run a number of simulations and save the respective trajectories
for i in range(100):
	# get a single realisation
	states, states2,times = SSA(S,X0,t_final,k)
	##a) save trajectory
	times =np.append(times,100.99)
	states= np.append(states,states[-1])
	states2= np.append(states2,states2[-1])
	X0_states = [states[0]]
	X1_states = [states2[0]]
	new_times = [0.0]
	X0_prev = -1
	X1_prev = -1
	cur = 1.0
	# print(times[-1])
	for j in range(len(times)):
		while times[j] >cur:
			new_times.append(cur)
			cur+=1
			X0_states.append(X0_prev)
			X1_states.append(X1_prev)
		if times[j]==cur:
			new_times.append(cur)
			cur+=1
			X0_states.append(states[j])
			X1_states.append(states2[j])
		X0_prev = states[j]
		X1_prev = states2[j]
	X0_stateses.append(X0_states)

means = []
sds = []
times = list(range(101))
# for i in X0_stateses:
# 	print(len(i))
from statistics import stdev, mean
for i in range(len(X0_stateses[0])):
	numbers = []
	for j in range(100):
		numbers.append(X0_stateses[j][i])
	means.append(mean(numbers))
	sds.append(stdev(numbers))

import matplotlib.pyplot as plt


means_plus = []
means_minus = []
for i in range(len(means)):
	means_plus.append(means[i]+sds[i])
	means_minus.append(means[i]-sds[i])

plt.plot(times,means,label="mean")
plt.plot(times,means_plus,label="mean+sd")
plt.plot(times,means_minus,label="mean-sd")
plt.xlabel("Time")
plt.ylabel("x1 value")
plt.title("Ex.3b")
plt.legend()
plt.show()


