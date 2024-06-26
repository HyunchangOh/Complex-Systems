####
# MCMC Exercise 4, Max von KLeist
####
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import scipy, math
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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
   
    # If negative -> redraw
    while theta1_p<0 or theta2_p<0:
        theta1_p = np.random.normal(theta1,sigma_p)     
        theta2_p = np.random.normal(theta2,sigma_p)   
    
    # # If negative -> skip
    # if theta1_p<0 or theta2_p<0:
    #     if i > burnin:
    #         Theta_s.append(theta)
    #     Acc_t.append(naccept)
    #     continue

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

np.savetxt('Project2b.txt',Theta_s,delimiter = ',',fmt='%1.2f');	


#### End of Metropolis Hastings ####
# comment out below for plots #
# Step 2: Marginal Empirical Distributions
data = np.array(Theta_s,ndmin=2)





# Step 2: Marginal Empirical Distributions
lambda_values = data[:, 0]
delta_values = data[:, 1]

# Create histograms for each parameter
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(lambda_values, bins=30, color='blue', alpha=0.7)
plt.title('Histogram of λ')
plt.xlabel('λ')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(delta_values, bins=30, color='green', alpha=0.7)
plt.title('Histogram of δ')
plt.xlabel('δ')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Step 3: Joint Distribution - Contour plot with line and scatter plot
plt.figure(figsize=(8, 6))

# Generate a kernel density estimate for the joint distribution
kde = gaussian_kde(data.T)

# Define the range for the contour plot
lambda_range = np.linspace(5,30, 100)
delta_range = np.linspace(0.05,0.30, 100)
Lambda, Delta = np.meshgrid(lambda_range, delta_range)
positions = np.vstack([Lambda.ravel(), Delta.ravel()])

# Evaluate the density at each point in the range
density = np.reshape(kde(positions).T, Lambda.shape)

# Plot the contour plot
plt.contour(Lambda, Delta, density, cmap='viridis', levels=10)

# Add a colorbar
plt.colorbar(label='Density')

# Plot the line (λ, 0.01λ)
line_lambda = np.linspace(5,30, 100)
line_delta = 0.01 * line_lambda
plt.plot(line_lambda, line_delta, color='red', label='(λ, 0.01λ)')

# Plot scatter plot
plt.scatter(lambda_values, delta_values, color='black', alpha=0.3, label='Data')

# Set y-axis limit based on delta quantiles
# delta_quantiles = np.percentile(delta_values, [5, 95])
plt.ylim(0.05,0.30)
plt.xlim(5,30)
plt.xlabel('λ')
plt.ylabel('δ')
plt.title('Joint Distribution of λ and δ')
plt.legend()

plt.show()