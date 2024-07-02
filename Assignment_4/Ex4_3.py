####
# MCMC Exercise 4, Max von KLeist
####

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy

#set seed
In = np.loadtxt('Input.txt')
np.random.seed(seed=int(In))

#set initial parameter estimates
ka = 0.5
ke = 0.3


# loads the input data
t, data = np.load('Data.npy')
#settings of the metropolis hastings algorithm
niters = 10000#
burnin = -1#np.round(0.9*niters)

#for evaluating the performance [optional]
naccept = 0
Acc_t = []

# model specifications
X0 = 200
X1 = 0


X0 = np.array([200,0])

#-----Fixed Quantities-----
# Stoichiometric matrix

# <------- fill in the model specifics ---->

#               r1   r2  r3 
S = np.array([  [-1,  0],#X1
                [1,  -1]])#X3

#reaction parameters
k = [0.5, 0.3]

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



#### Parameters for the priors, proposal and likelihood ####
# data for computing the priors
min1 = 0
max1 = 1
min2 = 0
max2 = 1

# data for the proposal distributions
sigma_p = 0.1

# parameters for likelihood
sigma_L = 10

#model  x(t) = 
def model(ka,ke,t,X0):
    sol = scipy.integrate.solve_ivp(RHS,[t[0],t[-1]],y0=X0,args=[[ka,ke]],vectorized=True,t_eval=t)
    return sol['y'][1]
  
#Likelihood   L = prod()
def likelihood(x,data,sigma):
    L = np.prod(stats.norm.pdf(data,loc=x,scale=sigma))
    return L

#prior distribution (for one parameter) v = 
def prior(min,max,y):  
    if y < min or y > max:
        return 0
    else:
        return 1 / (max - min)

#proposal distribution probability (for one parameter)
def proposal_prob(toParam,fromParam,sigma_p):
    p = stats.norm.pdf(np.log(toParam),loc= np.log(fromParam),scale =sigma_p)
    q = stats.norm.pdf(np.log(fromParam),loc= np.log(toParam),scale =sigma_p)
    if p!=q:
        print("yay")
    return p

### Begin of Metropolis Hastings Algorithm ###

# Initialize
# 2a. evaluate model at times t_i with parameters ka and ke
x = model(ka,ke,t,X0)

# 2b. compute likelihood
l = likelihood(x,data,sigma_L)
# 2b. compute priors
Prior_theta = prior(min1,max1,ka)*prior(min2,max2,ke)#

# Compute L * v (likelihood times prior)
posterior = Prior_theta * l

#store current parameter values
theta = [ka,ke]
Theta_s = []
Theta_s.append(theta)

###  Start iterating ###
for i in range(niters):
    # 1. propose parameters 
    ka_p = np.exp(np.random.normal(np.log(ka),sigma_p))                                 # <-----------
    ke_p = np.exp(np.random.normal(np.log(ke),sigma_p))
    # 1b. Compute Proposal distributions: 
    # Q(theta|theta')
    Q_Current_from_proposed = proposal_prob(ka,ka_p,sigma_p)*proposal_prob(ke,ke_p,sigma_p)
    #Q(theta'|theta)
    Q_Proposed_from_current = proposal_prob(ka_p,ka,sigma_p)*proposal_prob(ke_p,ke,sigma_p)
    assert proposal_prob(ka,ka_p,sigma_p) == proposal_prob(ka_p,ka,sigma_p), "Qs are different!!"
    # 2. Compute likelihood with proposed parameters
    # 2a. Evaluate model
    x_p = model(ka_p,ke_p,t,X0)
    # 2b. compute likelihood with proposed parameters
    l_p = likelihood(x_p,data,sigma_L) 
    # 2c. compute prior of proposed parameters
    Prior_theta_p = prior(min1,max1,ka_p) * prior(min2,max2,ke_p)
    
    # Compute L * v (likelihood times prior)
    posterior_p = Prior_theta_p * l_p

    #3. Compute acceptance criterion
    rho = min(1,posterior_p*Q_Current_from_proposed/(posterior*Q_Proposed_from_current))
    # 4.
    u = np.random.rand()
    
    # 5. Accept new parameters
    if u < rho:
        naccept += 1
        ka = ka_p
        ke = ke_p
        posterior = posterior_p
        theta = [ka,ke]
    # Store parameters only after burn-in phase
    if i > burnin:
        Theta_s.append(theta)
    Acc_t.append(naccept)
np.savetxt('Task3ParamEstimates.txt',Theta_s,delimiter = ',',fmt='%1.2f');	
#### End of Metropolis Hastings ####
# comment out below for plots #


# # a) Analyze efficiency    
# print("Efficiency = ", naccept/niters)

# ## Format output and plot
# Theta_s = np.array(Theta_s,ndmin=2)
# ka_s = Theta_s[:,0].flatten(order='F')
# ke_s = Theta_s[:,1].flatten(order='F')

# ##Histogram of parameter estimates (marginal distributions)
# fig, (ax0, ax1) = plt.subplots(nrows=2)
# ax0.hist(ka_s, bins=40, density = True)
# ax1.hist(ke_s, bins=40, density = True)

# ## Plot acceptance rate convergence 
# Acc_t = np.array(Acc_t)
# fig, ax_0 = plt.subplots(nrows=1)
# xvalues = range(niters)
# ax_0.plot(xvalues[1:],Acc_t[1:]/xvalues[1:],'k-o')
# plt.xlabel("iterations")
# plt.ylabel("acceptance rate")
# ax_0.set_yscale('log')

# ## Contour plot of parameter estimates
# NrGridPoints = 50
# xmin = ka_s.min()
# xmax = ka_s.max()
# ymin = ke_s.min()
# ymax = ke_s.max()
# X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([X.ravel(), Y.ravel()])
# values = np.vstack([ka_s,ke_s])
# kernel = stats.gaussian_kde(values)
# Z = np.reshape(kernel(positions).T, X.shape)
# fig, ax = plt.subplots()
# ax.contour(X, Y, Z)
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])
# plt.xlabel("ka")
# plt.ylabel("ke")
# plt.show()

# ## sanity check: prediction with  
# ka = np.median(ka_s)
# ke = np.median(ke_s)
# print(ka)
# print(ke)
# times = np.arange(t[0],t[-1],0.1)
# x = model(ka,ke,times,X0)
# plt.plot(times,x)
# plt.plot(times,data,'s--')
# plt.xlabel("time")
# plt.ylabel("concentration")
# plt.show()
