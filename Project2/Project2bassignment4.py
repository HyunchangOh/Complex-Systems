####
# MCMC Exercise 4, Max von KLeist
####
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy, math

#set seed
In = np.loadtxt('Input.txt')
np.random.seed(seed=int(In))

#set initial parameter estimates
theta1 = 30*np.random.rand()
theta2 = 3*np.random.rand()


Y = np.array([108,108,101,108,109,91,108,97,92,98])
t = 100

#settings of the metropolis hastings algorithm
niters = 100000#
burnin = np.round(0.9*niters)

#for evaluating the performance [optional]
naccept = 0
Acc_t = []

# data for the proposal distributions
sigma_p = 3

#model  x(t) = 
def model(l,d):
    return (l/d)*(1-np.exp(-d*t))

mu1 = 15
sigma1_sq = 5
# theta1 = lambda_
mu2 = 1
sigma2_sq = 10
# theta2 = delta_

#Likelihood   L = prod()
def likelihood(l,d):
    prod = 0
    x =model(l,d)
    for y in Y:
        for i in range(y):
            prod-=np.log(i+1)
        prod+=np.log(x)*y-x
    return -prod


#prior distribution (for one parameter) v = 
def prior(theta, mu,sigma_sq):
    exp_up = -(pow((theta-mu),2)/(2*sigma_sq))
    return (1/math.sqrt(2*sigma_sq*math.pi))*np.exp(exp_up)

#proposal distribution probability (for one parameter)
def proposal_prob(toParam,fromParam,sigma_p):
    p = stats.norm.pdf(np.log(toParam),loc= np.log(fromParam),scale =sigma_p)
    return p

### Begin of Metropolis Hastings Algorithm ###

# Initialize
# 2a. evaluate model at times t_i with parameters ka and ke 
# 2b. compute likelihood
l_a = likelihood(theta1,theta2)
# 2b. compute priors
Prior_theta = prior(theta1,mu1,sigma1_sq)*prior(theta2,mu2,sigma2_sq)#

# Compute L * v (likelihood times prior)
posterior = Prior_theta * l_a
#store current parameter values
theta = [theta1,theta2]
Theta_s = []
Theta_s.append(theta)

###  Start iterating ###
for i in range(niters):
    # 1. propose parameters 
    theta1_p = np.random.normal(theta1,sigma_p)     
    theta2_p = np.random.normal(theta2,sigma_p)                            # <-----------
    if theta1_p<0 or theta2_p<0:
        if i > burnin:
            Theta_s.append(theta)
        Acc_t.append(naccept)
        continue
    # 1b. Compute Proposal distributions: 
    # Q(theta|theta')
    Q_Current_from_proposed = proposal_prob(theta1,theta1_p,sigma_p)*proposal_prob(theta2,theta2_p,sigma_p)
    #Q(theta'|theta)
    Q_Proposed_from_current = proposal_prob(theta1_p,theta1,sigma_p)*proposal_prob(theta2_p,theta2,sigma_p)
    
    # 2. Compute likelihood with proposed parameters
    # 2a. Evaluate model
    # 2b. compute likelihood with proposed parameters
    l_p = likelihood(theta1_p,theta2_p) 
    # 2c. compute prior of proposed parameters
    Prior_theta_p = prior(theta1_p,mu1,sigma1_sq) * prior(theta2_p,mu2,sigma2_sq)
    
    # Compute L * v (likelihood times prior)
    posterior_p = Prior_theta_p * l_p

    #3. Compute acceptance criterion
    rho = min(1,(Q_Current_from_proposed/Q_Proposed_from_current)*(Prior_theta_p/Prior_theta)*np.exp(-l_p+l_a))
    # 4.
    u = np.random.rand()

    # 5. Accept new parameters
    if u < rho:
        naccept += 1
        theta1 = theta1_p
        theta2 = theta2_p
        posterior = posterior_p
        l_a = l_p
        theta = [theta1,theta2]
    # Store parameters only after burn-in phase
    if i > burnin:
        Theta_s.append(theta)
    Acc_t.append(naccept)
np.savetxt('Project2.txt',Theta_s,delimiter = ',',fmt='%1.2f');	
#### End of Metropolis Hastings ####
# comment out below for plots #



# # a) Analyze efficiency    
# print("Efficiency = ", naccept/niters)

## Format output and plot
Theta_s = np.array(Theta_s,ndmin=2)
lambda_s = Theta_s[:,0].flatten(order='F')
delta_s = Theta_s[:,1].flatten(order='F')

##Histogram of parameter estimates (marginal distributions)
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.hist(lambda_s, bins=40, density = True)
ax1.hist(delta_s, bins=40, density = True)

## Plot acceptance rate convergence 
Acc_t = np.array(Acc_t)
fig, ax_0 = plt.subplots(nrows=1)
xvalues = range(niters)
ax_0.plot(xvalues[1:],Acc_t[1:]/xvalues[1:],'k-o')
plt.xlabel("iterations")
plt.ylabel("acceptance rate")
ax_0.set_yscale('log')

## Contour plot of parameter estimates
NrGridPoints = 50
xmin = lambda_s.min()
xmax = lambda_s.max()
ymin = delta_s.min()
ymax = delta_s.max()
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([lambda_s,delta_s])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
fig, ax = plt.subplots()
ax.contour(X, Y, Z)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.xlabel("λ")
plt.ylabel("δ")
plt.show()

# ## sanity check: prediction with  
# ka = np.median(lambda_s)
# ke = np.median(delta_s)
# print(ka)
# print(ke)
# times = np.arange(t[0],t[-1],0.1)
# x = model(ka,ke,times,X0)
# plt.plot(times,x)
# plt.plot(times,data,'s--')
# plt.xlabel("time")
# plt.ylabel("concentration")
# plt.show()
