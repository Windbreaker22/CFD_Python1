import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Assign values and attributes; The parameters of ar, v, t, rho, p are all dimensionless 

nx = 30
dx = 3 / (nx)
x = np.linspace(0, 3, nx + 1)
c = 0.5
ar = np.zeros(nx + 1)
rho = np.zeros(nx + 1)
t = np.zeros(nx + 1)
v = np.zeros(nx + 1)
p = np.zeros(nx + 1)
ma = p.copy()
dt = 1
r = 1.4
nt = 2000
a = list()



# initial conditions for the flow variables

for i in range(nx + 1):

    ar[i] = 1 + 2.2 * (x[i] - 1.5)**2
    rho[i] = 1 - (0.3146 * x[i])
    t[i] = 1 - (0.2314 * x[i])
    v[i] = (0.1 + 1.09 * x[i]) * (t[i] ** 0.5)
    
    a = t[i] ** 0.5
    temp = c * (dx / (a + v[i])) 
    
    if temp < dt :
       dt = temp

v_temp = v.copy()
rho_temp = rho.copy()
t_temp = t.copy()

pdrho = np.zeros(nx + 1)
pdv = np.zeros(nx + 1)
pdte = np.zeros(nx + 1)

cdrho = np.zeros(nx + 1)
cdv = np.zeros(nx + 1)
cdte = np.zeros(nx + 1)

drho = np.zeros(nx + 1)
dv = np.zeros(nx + 1)
dte = np.zeros(nx + 1)

pred_rho = np.zeros(nx + 1)
pred_v = np.zeros(nx + 1)
pred_t = np.zeros(nx + 1)

v_throat = np.zeros(nt)
rho_throat = np.zeros(nt)
t_throat = np.zeros(nt)
ma_throat = np.zeros(nt)

for j in range(nt):
    
    for i in range(1, nx):
    
        # Mccormack method and its predictor step
        # The spatial rate of changes are forward moving
        # pdrho , pdv and pdte are the rate of change of flow variables at i grid point at time step t
    
        pdrho[i] = - (rho[i] * ( (v[i+1] - v[i]) / dx ))  - (rho[i] * v[i] * ( (np.log(ar[i+1]) - np.log(ar[i])) / dx)) - (v[i] * (rho[i+1]- rho[i]) / dx)
        
        pdv[i] =  - (v[i] * ((v[i+1] - v[i]) / dx)) - ( (((t[i+1] - t[i])/dx) + (t[i] * ((rho[i+1] - rho[i])/dx) / rho[i])) / r)
    
        pdte[i] = - (v[i] * ((t[i+1] - t[i])/dx)) - ((r-1) * t[i] * ( ((v[i+1] - v[i]) / dx) + (v[i] * ((np.log(ar[i+1]) - np.log(ar[i])) / dx)) ))

        # Here we find out the predicted values of rho, v and temp at time step t+ dt with the help of the data obtained in the expressions written above
        pred_rho[i] = rho[i] + (pdrho[i] * dt)
        pred_v[i] = v[i] + (pdv[i] * dt)
        pred_t[i] = t[i] + (pdte[i]* dt)
    
    #Boundary conditions for the predictor parameters based on linear approximation
    pred_v[0] = 2 * pred_v[1] - pred_v[2]
    pred_t[0] = 1
    pred_rho[0] = 1
    
    pred_v[-1] = 2 * pred_v[-2] - pred_v[-3]
    pred_rho[-1] = 2 * pred_rho[-2] - pred_rho[-3]
    pred_t[-1] = 2 * pred_t[-2] - pred_t[-3]
    
    for i in range(1,nx) :

        # Mccormack method and its corrector step can be calculated for internal grid points
        # the spatial rate of changes are rearwards
        # cdrho, cdv and cdte are the rate of change of flow variables at predicted values on the time step t + dt   
    
        cdrho[i] = - (pred_rho[i] * ( (pred_v[i] - pred_v[i-1]) / dx ))  - (pred_rho[i] * pred_v[i] * ( (np.log(ar[i]) - np.log(ar[i-1])) / dx)) - (pred_v[i] * (pred_rho[i]- pred_rho[i-1]) / dx)
        
        cdv[i] =  - (pred_v[i] * ((pred_v[i] - pred_v[i-1]) / dx)) - ( (((pred_t[i] - pred_t[i-1])/dx) + (pred_t[i] * ((pred_rho[i] - pred_rho[i-1])/dx) / pred_rho[i])) / r)
        
        cdte[i] = - (pred_v[i] * ((pred_t[i] - pred_t[i-1])/dx)) - ((r-1) * pred_t[i] * ( ((pred_v[i] - pred_v[i-1]) / dx) + (pred_v[i] * ((np.log(ar[i]) - np.log(ar[i-1])) / dx)) ))
        
        
        # Average values of predicted and corrected rate of change of flow variables is written below
        drho[i] = 0.5 * (pdrho[i] + cdrho[i])
        dv[i] = 0.5 * (pdv[i] + cdv[i])
        dte[i] = 0.5 * (pdte[i] + cdte[i])
            
        # New values of rho, velocity & temp at the time step t + dt
    for i in range(nx + 1):
        
        v[i] = v[i] + (dv[i] * dt)
        rho[i] = rho[i] + (drho[i] * dt)
        t[i] = t[i] + (dte[i] * dt)
        p[i] = rho[i] * t[i]
        ma[i] = v[i] / (t[i] ** 0.5)
        
        
    # Boundary conditions are calculated properly using a linear approximation
    v[0] = 2 * v[1] - v[2]
    rho[0] = 1
    t[0] = 1        
    
    v[-1] = 2 * v[-2] - v[-3]
    rho[-1] = 2 * rho[-2] - rho[-3]
    t[-1] = 2 * t[-2] - t[-3]
        
    rho_throat[j] = rho[15]
    v_throat[j] = v[15]
    t_throat[j] = t[15]
    ma_throat[j] = ma[15]   
    
#data = {'point': x, 'velocity' : v, 'density' : rho, 'temperature' : t, 'Local pressure': p, 'Mach number' : ma}
#df = pd.DataFrame(data)
#print(df)

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