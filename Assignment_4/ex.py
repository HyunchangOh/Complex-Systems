import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the input data
t, data = np.load('Data.npy')

# Set the seed
In = np.loadtxt('Input.txt')
np.random.seed(seed=int(In))

# Set initial parameter estimates
ka = 0.5
ke = 0.3

# Settings of the Metropolis-Hastings algorithm
niters = 10
burnin = -1#np.round(0.9 * niters)

# For evaluating the performance [optional]
naccept = 0
Acc_t = []

# Model specifications
X0 = 200

# Parameters for the priors, proposal, and likelihood
min1 = 0
max1 = 1
min2 = 0
max2 = 1
sigma_p = 0.1
sigma_L = 10

# Model function
def model(ka, ke, t, X0):
    x = X0 - ka * t
    return x

# Likelihood function
def likelihood(x, data, sigma):
    L = np.prod(stats.norm.pdf(data, loc=x, scale=sigma))
    return L

# Prior distribution (for one parameter)
def prior(min_val, max_val, y):
    if y < min_val or y > max_val:
        return 0
    else:
        return 1 / (max_val - min_val)

# Proposal distribution probability (for one parameter)
def proposal_prob(to_param, from_param, sigma_p):
    p = stats.norm.pdf(np.log(to_param), loc=np.log(from_param), scale=sigma_p)
    return p

# Initialize
x = model(ka, ke, t, X0)
l = likelihood(x, data, sigma_L)
Prior_theta = prior(min1, max1, ka) * prior(min2, max2, ke)
posterior = Prior_theta * l
theta = [ka, ke]
Theta_s = []
Theta_s.append(theta)

# Start iterating
for i in range(niters):
    ka_p = np.random.lognormal(np.log(ka), sigma_p)
    ke_p = np.random.lognormal(np.log(ke), sigma_p)
    
    Q_Current_from_proposed = proposal_prob(ka, ka_p, sigma_p) * proposal_prob(ke, ke_p, sigma_p)
    Q_Proposed_from_current = proposal_prob(ka_p, ka, sigma_p) * proposal_prob(ke_p, ke, sigma_p)
    
    x_p = model(ka_p, ke_p, t, X0)
    l_p = likelihood(x_p, data, sigma_L)
    Prior_theta_p = prior(min1, max1, ka_p) * prior(min2, max2, ke_p)
    posterior_p = Prior_theta_p * l_p

    rho = min(1, posterior_p * Q_Current_from_proposed / (posterior * Q_Proposed_from_current))
    u = np.random.rand()

    if u < rho:
        naccept += 1
        ka = ka_p
        ke = ke_p
        posterior = posterior_p
        theta = [ka, ke]
    
    if i > burnin:
        Theta_s.append(theta)
    Acc_t.append(naccept)

np.savetxt('Task3ParamEstimates.txt', Theta_s, delimiter=',', fmt='%1.2f')
print("Accepted parameter values:", Theta_s)