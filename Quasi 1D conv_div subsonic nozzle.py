# A subsonic convergent - divergent nozzle's speed (Mach number) not only depends on the Area ratio..
# .. but also on the exit pressure/stag. pressure ratio.

# Here at the influx BC, two properties will be specified and one property (velocity) will be floated
# At the outflux BC, one property (exit pressure) will be specified, and two properties will be floated

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#Attribute definition

nx = 30 # nx represents the number of grid point intervals
dx = 3 / (nx) # dx represents the distance between grid points

c = 0.5 # CFL number for stability; for linear (strictly) and non linear Hyperbolic PDE, it should be less than 1
nt = 5000 # Number of time steps
r = 1.4 # Specific heat ratio
epr = 0.93 # Exit pressure ratio

x = np.linspace(0,3, nx + 1) # creates 31 points along the flow 
temp = np.zeros(nx + 1) 
rho = np.zeros(nx + 1)
ar = np.zeros(nx + 1)
t = np.zeros(nx + 1)
v = np.zeros(nx + 1) 
ma = np.zeros(nx + 1)
p = np.zeros(nx + 1)

# Defining initial conditions of Area, density, temp & Velocity
for i in range(nx + 1): 
    
    if x[i] < 1.5 :
        ar[i] = 1 + 2.2 * (x[i] - 1.5) ** 2
    
    else : 
        ar[i] = 1 + 0.2223 * ( x[i] - 1.5) ** 2
        
    rho[i] = 1 - 0.023 * x[i]
    t[i] = 1 - 0.009333 * x[i]
    v[i] = 0.05 + 0.11 * x[i]
    temp[i] = c * dx / (t[i]**0.5 + v[i])
    
dt = min(temp)

# For plotting purposes,

v_temp, rho_temp, t_temp = v.copy(), rho.copy(), t.copy()

v_throat = np.zeros(nt)
rho_throat = np.zeros(nt)
t_throat = np.zeros(nt)
p_throat = np.zeros(nt)
ma_throat = np.zeros(nt)

# Predictor step of Mccormack's method of explict FD method using forward spatial differences

pdrho = np.zeros(nx + 1)
pdv = np.zeros(nx + 1)
pdte = np.zeros(nx + 1)

pred_v = np.zeros(nx + 1)
pred_rho = np.zeros(nx + 1)
pred_t = np.zeros(nx + 1)

cdrho = np.zeros(nx + 1)
cdv = np.zeros(nx + 1)
cdte = np.zeros(nx + 1)

for j in range(nt):
    
    for i in range(1, nx) :
    
        pdrho[i] = - (rho[i] * ( (v[i+1] - v[i]) / dx ))  - (rho[i] * v[i] * ( (np.log(ar[i+1]) - np.log(ar[i])) / dx)) - (v[i] * (rho[i+1]- rho[i]) / dx)
        
        pdv[i] =  - (v[i] * ((v[i+1] - v[i]) / dx)) - ( (((t[i+1] - t[i])/dx) + (t[i] * ((rho[i+1] - rho[i])/dx) / rho[i])) / r)
    
        pdte[i] = - (v[i] * ((t[i+1] - t[i])/dx)) - ((r-1) * t[i] * ( ((v[i+1] - v[i]) / dx) + (v[i] * ((np.log(ar[i+1]) - np.log(ar[i])) / dx)) ))
    
        #Predicted values for time step t + dt are calculated with the help of above expressions
        
        pred_rho[i] = rho[i] + (pdrho[i] * dt)
        pred_v[i] = v[i] + (pdv[i] * dt)
        pred_t[i] = t[i] + (pdte[i]* dt)
        
    #Boundary conditions for the predictor parameters based on linear approximation
    pred_v[0] = 2 * pred_v[1] - pred_v[2]
    pred_t[0] = 1
    pred_rho[0] = 1
    
    pred_v[-1] = 2 * pred_v[-2] - pred_v[-3]
    pred_t[-1] = 2 * pred_t[-2] - pred_t[-3]
    pred_rho[-1] = epr / pred_t[-1]
    
    for i in range(1,nx) :

        # Mccormack method and its corrector step can be calculated for internal grid points where the spatial rate of changes are rearwards
        # cdrho, cdv and cdte are the rate of change of flow variables at predicted values on the time step t + dt   
    
        cdrho[i] = - (pred_rho[i] * ( (pred_v[i] - pred_v[i-1]) / dx ))  - (pred_rho[i] * pred_v[i] * ( (np.log(ar[i]) - np.log(ar[i-1])) / dx)) - (pred_v[i] * (pred_rho[i]- pred_rho[i-1]) / dx)
        
        cdv[i] =  - (pred_v[i] * ((pred_v[i] - pred_v[i-1]) / dx)) - ( (((pred_t[i] - pred_t[i-1])/dx) + (pred_t[i] * ((pred_rho[i] - pred_rho[i-1])/dx) / pred_rho[i])) / r)
        
        cdte[i] = - (pred_v[i] * ((pred_t[i] - pred_t[i-1])/dx)) - ((r-1) * pred_t[i] * ( ((pred_v[i] - pred_v[i-1]) / dx) + (pred_v[i] * ((np.log(ar[i]) - np.log(ar[i-1])) / dx)) ))
        
    
        # New values of rho, velocity & temp at the time step t + dt is calculated by taking averages of corrector & Predictor rates of changes
    for i in range(nx + 1):
        
        v[i] = v[i] + (0.5 * (pdv[i] + cdv[i]) * dt)
        rho[i] = rho[i] + (0.5 * (pdrho[i] + cdrho[i]) * dt)
        t[i] = t[i] + (0.5 * (pdte[i] + cdte[i]) * dt)
        ma[i] = v[i] / (t[i] ** 0.5)
        p[i] = rho[i]/t[i]
        
     # Boundary conditions are calculated properly using a linear approximations and specifications
     
    v[0] = 2 * v[1] - v[2]
    rho[0] = 1
    t[0] = 1        
     
    v[-1] = 2 * v[-2] - v[-3]
    t[-1] = 2 * t[-2] - t[-3]
    rho[-1] = epr / t[-1]   
        
    rho_throat[j] = rho[15]
    v_throat[j] = v[15]
    t_throat[j] = t[15]
    ma_throat[j] = ma[15] 
    p_throat[j] = p[15] 

plt.plot(np.linspace(0,3,nx + 1), v_temp, label = "Initial condition")
plt.plot(np.linspace(0, 3, nx + 1), v, label = "After given time steps")
plt.legend()
plt.title('Velocity variation')
plt.show()

plt.plot(np.linspace(0,3,nx + 1), rho_temp, label = "Initial condition")
plt.plot(np.linspace(0, 3, nx + 1), rho, label = "After given time steps")
plt.legend()
plt.title('Density variation')
plt.show()

plt.plot(np.linspace(0,3,nx + 1), t_temp, label = "Initial condition")
plt.plot(np.linspace(0, 3, nx + 1), t, label = "After given time steps")
plt.legend()
plt.title('Temperature variation')
plt.show()

plt.plot(rho_throat, label = "Density at throat")
plt.legend()
plt.show()

plt.plot(t_throat, label = "Temperature at throat")
plt.legend()
plt.show()

plt.plot(v_throat, label = "Velocity at throat")
plt.legend()
plt.show()

plt.plot(ma_throat, label = "Mach Number at throat")
plt.legend()
plt.show()

plt.plot(x,p, label = "Pressure variation")
plt.title("Pressue variation")
plt.show()

data = {'point': x, 'velocity' : v, 'density' : rho, 'temperature' : t, 'Local pressure': p, 'Mach number' : ma}
df = pd.DataFrame(data)
print(df)