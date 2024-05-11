import numpy as np
import math
import matplotlib.pyplot as plt
Ys = np.array([108,108,101,108,109,91,108,97,92,98])

t = 100

def x(t,l,d):
    return l/d*(1-np.exp(-d*t))

def likelihood(Y,l,d,t):
    prod = 0
    for y in Y:
        x_i =x(t,l,d)
        for i in range(y):
            prod+=np.log(i+1)
        prod-=np.log(pow(x_i,y)*np.exp(-x_i))
    return prod


l_range = np.linspace(5, 30, 100)
d_range = np.linspace(0.05, 0.3, 100)

L, D = np.meshgrid(l_range, d_range)


likelihood_values = likelihood(Ys, L, D, t)

levels = np.arange(30, 50, 2)
contour_plot = plt.contour(L, D, likelihood_values, levels=levels)
plt.clabel(contour_plot, inline=True, fontsize=8)


line_l = np.linspace(5, 30, 100)
line_d = 0.01 * line_l
plt.plot(line_l, line_d, color='red', label='(l, 0.01 * l)')


plt.xlabel('l')
plt.ylabel('d')
plt.legend()

# Show the plot
plt.title('Likelihood')
plt.grid(True)
plt.show()
plt.clf()

#####################################################################################
#################################### PART II ########################################
#####################################################################################

mu1 = 15
sigma1_sq = 5
# theta1 = lambda_
mu2 = 1
sigma2_sq = 10
# theta2 = delta_

def prior(theta, mu,sigma_sq):
    exp_up = -(((theta-mu)**2)/(2*sigma_sq))
    return (1/math.sqrt(2*sigma_sq*math.pi))*np.exp(exp_up)


l_range = np.linspace(5, 30, 100)
d_range = np.linspace(0.05, 0.3, 100)

L2, D2 = np.meshgrid(l_range, d_range)

def posterior(Y,l,d,t):
    prod = 0
    for y in Y:
        x_i =x(t,l,d)
        for i in range(y):
            prod+=np.log(i+1)
        prod-=np.log(pow(x_i,y)*np.exp(-x_i))
    prod -= prior(l,mu1,sigma1_sq)
    prod -= prior(d,mu2,sigma2_sq)
    return prod

posterior_values = posterior(Ys, L, D, t)

levels = np.arange(30, 50, 2)
contour_plot = plt.contour(L, D, posterior_values, levels=levels)
plt.clabel(contour_plot, inline=True, fontsize=8)


line_l = np.linspace(5, 30, 100)
line_d = 0.01 * line_l
plt.plot(line_l, line_d, color='red', label='(l, 0.01 * l)')


plt.xlabel('l')
plt.ylabel('d')
plt.legend()

# Show the plot
plt.title('Unnormalized Posterior with Gaussian Priors')
plt.grid(True)
plt.show()
plt.clf()