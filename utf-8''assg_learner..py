#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# -----
# 
# In this assignment you will recursively estimate the position of a vehicle along a trajectory using available measurements and a motion model. 
# 
# The vehicle is equipped with a very simple type of LIDAR sensor, which returns range and bearing measurements corresponding to individual landmarks in the environment. The global positions of the landmarks are assumed to be known beforehand. We will also assume known data association, that is, which measurment belong to which landmark.
# 
# ## Motion and Measurement Models
# -----
# 
# ### Motion Model
# 
# The vehicle motion model recieves linear and angular velocity odometry readings as inputs, and outputs the state (i.e., the 2D pose) of the vehicle:
# 
# \begin{align}
# \mathbf{x}_{k} &= \mathbf{x}_{k-1} + T
# \begin{bmatrix}
# \cos\theta_{k-1} &0 \\
# \sin\theta_{k-1} &0 \\
# 0 &1
# \end{bmatrix}
# \left(
# \begin{bmatrix}
# v_k \\
# \omega_k
# \end{bmatrix}
# + \mathbf{w}_k
# \right)
# \, , \, \, \, \, \, \mathbf{w}_k = \mathcal{N}\left(\mathbf{0}, \mathbf{Q}\right)
# \end{align}
# 
# - $\mathbf{x}_k = \left[ x \, y \, \theta \right]^T$ is the current 2D pose of the vehicle
# - $v_k$ and $\omega_k$ are the linear and angular velocity odometry readings, which we use as inputs to the model
# 
# The process noise $\mathbf{w}_k$ has a (zero mean) normal distribution with a constant covariance $\mathbf{Q}$.
# 
# ### Measurement Model
# 
# The measurement model relates the current pose of the vehicle to the LIDAR range and bearing measurements $\mathbf{y}^l_k = \left[r \, \phi \right]^T$.
# 
# \begin{align}
# \mathbf{y}^l_k =
# \begin{bmatrix}
# \sqrt{(x_l - x_k - d\cos\theta_{k})^2 + (y_l - y_k - d\sin\theta_{k})^2} \\
# atan2\left(y_l - y_k - d\sin\theta_{k},x_l - x_k - d\cos\theta_{k}\right) - \theta_k
# \end{bmatrix}
# +
# \mathbf{n}^l_k
# \, , \, \, \, \, \, \mathbf{n}^l_k = \mathcal{N}\left(\mathbf{0}, \mathbf{R}\right)
# \end{align}
# 
# - $x_l$ and $y_l$ are the ground truth coordinates of the landmark $l$
# - $x_k$ and $y_k$ and $\theta_{k}$ represent the current pose of the vehicle
# - $d$ is the known distance between robot center and laser rangefinder (LIDAR)
# 
# The landmark measurement noise $\mathbf{n}^l_k$ has a (zero mean) normal distribution with a constant covariance $\mathbf{R}$.
# 
# ## Getting Started
# -----
# 
# Since the models above are nonlinear, we recommend using the extended Kalman filter (EKF) as the state estimator.
# Specifically, you will need to provide code implementing the following steps:
# - the prediction step, which uses odometry measurements and the motion model to produce a state and covariance estimate at a given timestep, and
# - the correction step, which uses the range and bearing measurements provided by the LIDAR to correct the pose and pose covariance estimates
# 
# ### Unpack the Data
# First, let's unpack the available data:

# In[1]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]
#print(om)
# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]
print(b)


# Note that distance from the LIDAR frame to the robot center is provided and loaded as an array into the `d` variable.
# 
# ### Ground Truth
# If available, it is useful to plot the ground truth position and orientation before starting the assignment.
# 
# <table><tr>
# <td> <img src="data/gtruth.png" alt="Ground Truth" width="350"/> </td>
# <td> <img src="data/gtruth2.png" alt="Ground Truth" width="350"/> </td>
# </tr></table>
# 
# Notice that the orientation values are wrapped to the $\left[-\pi,\pi\right]$ range in radians.

# ### Initializing Parameters
# 
# Now that our data is loaded, we can begin getting things set up for our solver. One of the
# most important aspects of designing a filter is determining the input and measurement noise covariance matrices, as well as the initial state and covariance values. We set the values here:

# In[2]:


#v_var = 0.025# translation velocity variance  
#om_var = 0.95 # rotational velocity variance 
#r_var = 0.01  # range measurements variance
#b_var = 0.1 # bearing measurement variance

v_var = 0.1# translation velocity variance  
om_var = 0.15 # rotational velocity variance 
r_var = 0.01  # range measurements variance
b_var = 0.008

Q_km = np.diag([v_var, om_var]) # input noise covariance 
print(Q_km)
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
#print(x_est[0])
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance


# **Remember:** that it is neccessary to tune the measurement noise variances `r_var`, `b_var` in order for the filter to perform well!
# 
# In order for the orientation estimates to coincide with the bearing measurements, it is also neccessary to wrap all estimated $\theta$ values to the $(-\pi , \pi]$ range.

# In[3]:


# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x


# 
# ## Correction Step
# -----
# First, let's implement the measurement update function, which takes an available landmark measurement $l$ and updates the current state estimate $\mathbf{\check{x}}_k$.
# For each landmark measurement received at a given timestep $k$, you should implement the following steps:
# 
# - Compute the measurement model Jacobians at $\mathbf{\check{x}}_{k}$
# \begin{align}
# \mathbf{y}^l_k = &\mathbf{h}(\mathbf{x}_{k}, \mathbf{n}^l_k) \\\\
# \mathbf{H}_{k} = \frac{\partial \mathbf{h}}{\partial \mathbf{x}_{k}}\bigg|_{\mathbf{\check{x}}_{k},0}& \, , \, \, \, \,
# \mathbf{M}_{k} = \frac{\partial \mathbf{h}}{\partial \mathbf{n}_{k}}\bigg|_{\mathbf{\check{x}}_{k},0} \, .
# \end{align}
# - Compute the Kalman Gain
# \begin{align}
# \mathbf{K}_k &= \mathbf{\check{P}}_k \mathbf{H}_k^T \left(\mathbf{H}_k \mathbf{\check{P}}_k \mathbf{H}_k^T + \mathbf{M}_k \mathbf{R}_k \mathbf{M}_k^T \right)^{-1} 
# \end{align}
# - Correct the predicted state
# \begin{align}
# \mathbf{\check{y}}^l_k &= \mathbf{h}\left(\mathbf{\check{x}}_k, \mathbf{0}\right) \\
# \mathbf{\hat{x}}_k &= \mathbf{\check{x}}_k + \mathbf{K}_k \left(\mathbf{y}^l_k - \mathbf{\check{y}}^l_k\right)
# \end{align}
# - Correct the covariance
# \begin{align}
# \mathbf{\hat{P}}_k &= \left(\mathbf{I} - \mathbf{K}_k \mathbf{H}_k \right)\mathbf{\check{P}}_k
# \end{align}

# In[20]:


def measurement_update(lk, rk, bk, P_check, x_check):
    x1 = lk[0]
    y1 = lk[1]
    x_prime=x_check.reshape([3,1])
    xk = x_prime[0][0]
    yk = x_prime[1][0]
    print("88888")
    print(x_prime)
    theta=wraptopi(x_prime[2][0])
    print(theta)
    y=np.zeros([2,1])
    y[0][0] = rk
    y[1][0] = wraptopi(bk)
    #print(y)
    # 1. Compute measurement Jacobian
    den = (x1-xk-d*np.cos(theta))**2 + (y1-yk-d*np.sin(theta))**2
    hk = np.zeros([2,3])
    hk[0][0] = -1*(x1-xk-d*np.cos(theta))/np.sqrt(den)
    #print(hk[0][0])
    hk[0][1] = -1*(y1-yk-d*np.sin(theta))/np.sqrt(den)
    #print(hk[0][1])
    
    hk[0][2] = (d*(np.sin(theta)*(x1-xk) - np.cos(theta)*(y1-yk)))/np.sqrt(den)
    hk[1][0] = (-1*(y1-yk-d*np.sin(theta)))/den
    hk[1][1] = (-1*(x1-xk-d*np.cos(theta)))/den
    hk[1][2] = ((d*d-1*d*(np.cos(theta)*(x1-xk) +np.sin(theta)*(y1-yk)))/den) -1
    p = np.arctan2(hk[0][1],hk[0][0])-theta
    p=wraptopi(p)
    #print(p)
    yk = np.zeros([2,1])
    #print(eq)
    yk[0][0] = np.sqrt(den)
    yk[1][0] = p
    #print(yk)
    mk = np.identity(2)
    #print(mk)
    
    # 2. Compute Kalman Gain
    #print(hk@P_check@hk.T + mk@cov_y@mk.T)
    
    inverse = np.linalg.inv(hk@P_check@hk.T + cov_y)
    #print(inverse)
    kg = P_check@hk.T@inverse
    #print(kg)
    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    print(kg)
    diff=(y-yk)
    diff[1][0]=wraptopi(diff[1][0])
    print(kg@diff)
    
    x_prime = x_prime + kg@diff
    
    x_prime[2][0] = wraptopi(x_prime[2][0])
    x_check = x_prime.reshape(3)
    
    # 4. Correct covariance
    
    iden =np.identity(3)
    P_check = (iden - kg@hk)@P_check
    #print(P_check)
    #print(x_check)
    return x_check, P_check


# ## Prediction Step
# -----
# Now, implement the main filter loop, defining the prediction step of the EKF using the motion model provided:
# 
# \begin{align}
# \mathbf{\check{x}}_k &= \mathbf{f}\left(\mathbf{\hat{x}}_{k-1}, \mathbf{u}_{k-1}, \mathbf{0} \right) \\
# \mathbf{\check{P}}_k &= \mathbf{F}_{k-1}\mathbf{\hat{P}}_{k-1}\mathbf{F}_{k-1}^T + \mathbf{L}_{k-1}\mathbf{Q}_{k-1}\mathbf{L}_{k-1}^T \, .
# \end{align}
# 
# Where
# 
# \begin{align}
# \mathbf{F}_{k-1} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}_{k-1}}\bigg|_{\mathbf{\hat{x}}_{k-1},\mathbf{u}_{k},0}  \, , \, \, \, \,
# \mathbf{L}_{k-1} = \frac{\partial \mathbf{f}}{\partial \mathbf{w}_{k}}\bigg|_{\mathbf{\hat{x}}_{k-1},\mathbf{u}_{k},0} \, .
# \end{align}

# In[21]:


#### 5. Main Filter Loop #######################################################################
x_check = np.zeros(3)
x_check=x_est[0]
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    #print(delta_t)

    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    print(k-1)
    x_check=x_est[k-1]
    
    u = np.zeros([2,1])
    u[0][0]=v[k-1]
    u[1][0]=om[k-1]
    
    L_k = np.zeros([3, 2])
    
    x_p = x_check.reshape([3,1])
    x_p[2][0] = wraptopi(x_p[2][0])
    xp = x_p[0][0]
    yp = x_p[1][0]
    thetap = wraptopi(x_p[2][0])
    L_k[0][0] = np.cos(thetap)
    L_k[0][1] = 0.
    L_k[1][0] = np.sin(thetap)
    L_k[1][1] = 0.
    L_k[2][0] = 0.
    L_k[2][1] = 1.
    
    print(x_p)
    sushlok = np.zeros([3, 1])
    sushlok = L_k@u
    sushlok[2][0] = wraptopi(sushlok[2][0])
    x_p = x_p + delta_t*(sushlok)
    print(delta_t*(L_k@u))
    print(x_p)
    x_p[2][0] = wraptopi(x_p[2][0])
    print(x_p)
    x_check = x_p.reshape(3)
    
    # 2. Motion model jacobian with respect to last state
    
    F_km = np.zeros([3, 3])
    F_km[0][0] = 1.
    F_km[0][1] = 0.
    F_km[0][2] = -1*v[k-1]*np.sin(thetap)*delta_t
    F_km[1][0] = 0
    F_km[1][1] = 1.
    F_km[1][2] = v[k-1]*np.cos(thetap)*delta_t
    F_km[2][0] = 0.
    F_km[2][1] = 0.
    F_km[2][2] = 1.
    
    # 3. Motion model jacobian with respect to noise
    
    L_km = np.zeros([3, 2])
    L_km[0][0] = np.cos(thetap)*delta_t
    L_km[0][1] = 0.
    L_km[1][0] = np.sin(thetap)*delta_t
    L_km[1][1] = 0.
    L_km[2][0] = 0.
    L_km[2][1] = 1.*delta_t
    
    
    
    # 4. Propagate uncertainty
    
    P_check = np.zeros([3,3])
    P_check = P_est[k-1, :, :]
    
    P_check = F_km@P_check@F_km.T + L_km@Q_km@L_km.T
    # 5. Update state estimate using available landmark measurements
    
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    print(x_check)
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check


# Let's plot the resulting state estimates:

# In[22]:



e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

print(x_est.shape[0])
temp=np.zeros(x_est.shape[0])
for i in range(x_est.shape[0]):
    temp[i] = -1*x_est[x_est.shape[0]-i-1][2]

    
e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], temp[:])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()


# Are you satisfied wth your results? The resulting trajectory should closely resemble the ground truth, with minor "jumps" in the orientation estimate due to angle wrapping. If this is the case, run the code below to produce your solution file.

# In[69]:


with open('submission.pkl', 'wb') as f:
    pickle.dump(x_est, f, pickle.HIGHEST_PROTOCOL)


# In[ ]:




