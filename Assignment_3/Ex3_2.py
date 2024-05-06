# Import required modules 
import numpy as np 
import matplotlib.pyplot as plt 

# Meshgrid 
x, y = np.meshgrid(np.linspace(-15, 15, 10), 
				np.linspace(-15, 15, 10)) 

# Directional vectors 
x1 = 1 - 0.01*x*y
x2 = 1 - 0.01*x*y

# Plotting Vector Field with QUIVER 
plt.quiver(x, y, x1, x2, color='g') 
plt.title('Vector Field') 

# Setting x, y boundary limits 
plt.xlim(-15, 15) 
plt.ylim(-15, 15) 

# Show plot with grid 
plt.grid() 
plt.show() 
