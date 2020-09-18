# Starter code for the Coursera SDC Course 2 final project.
#
# Author: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']
print(lidar.t)
################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()

################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can insert it into our solver as another position measurement,
# just as we do for GNSS. However, the LIDAR frame is not the same as the frame shared by the
# IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).

C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])
"""
# Incorrect calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.05).
C_li = np.array([
      [ 0.9975 , -0.04742,  0.05235],
      [ 0.04992,  0.99763, -0.04742],
      [-0.04998,  0.04992,  0.9975 ]
 ])
"""
t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.50
var_imu_w = 0.25
var_gnss  = 0.01
var_lidar = 2.00

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i  = 0
lidar_i = 0

#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain
    kg = p_cov_check@h_jac.T@np.linalg.inv(h_jac@p_cov_check@h_jac.T + sensor_var)
    # 3.2 Compute error state
    error = kg@(y_k - p_check)
    # 3.3 Correct predicted state
    p_hat = np.zeros(p_check.shape)
    v_hat = np.zeros(v_check.shape)
    q_hat = np.zeros(q_check.shape)
    p_cov_hat = np.zeros(p_cov_check.shape)
    delta_theta = np.zeros(p_check.shape)
    for i in range(3):
        p_hat[i] = p_check[i] + error[i]

    for i in range(3):
        v_hat[i] = v_check[i] + error[i+3]

    for i in range(3):
        delta_theta[i] = delta_theta[i] + error[i+6]

    quaterr = Quaternion(axis_angle = delta_theta).to_numpy()
    #print(quaterr)
    q_hat = (Quaternion(*quaterr).quat_mult_left(Quaternion(*q_check), out='np'))
    q_hat = (Quaternion(*q_hat).normalize()).to_numpy()
    # 3.4 Compute corrected covariance
    p_cov_hat = (np.eye(9) - kg@h_jac)@p_cov_check

    return p_hat, v_hat, q_hat, p_cov_hat

#### 5. Main Filter Loop #######################################################################
def cholesky(A):
    root = np.zeros(A.shape())
    root[0][0] = ((A[0][0])**0.5)
    root[1][0] = A[1][0]/root[0][0]
    root[2][0] = A[2][0]/root[0][0]
    root[1][1] = (A[1][1]-(root[1][0]**2))**0.5
    root[2][1] = (A[2][1] -root[2][0]*root[1][0]) /root[1][1]
    root[2][2] = (A[2][2]-root[2][0]**2-root[2][1]**2)**0.5
    return root

def sigmapoint(u,p,l):
    sp = p.shape
    #print(sp)
    n = sp[0]
    scale = (l + n)**0.5
    m = 2*sp[0] + 1
    sigma = np.zeros([n,m])
    u = u.reshape(n)
    sigma[:,0] = u
    for i in range(n):
        sigma[:,i+1] = u + scale*p[:,i]

    for j in range(n):
        sigma[:,i+n+1] = u - scale*p[:,i]
    return sigma




################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]
    q_est[k-1] = (Quaternion(*q_est[k-1]).normalize()).to_numpy()
    eular = Quaternion(*q_est[k-1]).to_euler()
    #print(eular)
    et = np.zeros(eular.shape)
    u = np.zeros([9,1])
    n = 9
    print("pest")
    print(p_est[k-1])
    u[0:3,0] = p_est[k-1][0:3]
    u[3:6,0] = v_est[k-1][0:3]
    u[6:9,0] = eular
    k = 1
    alpha = 0.5
    limbda = (alpha**2)*(n + k) - n
    sigma1 = sigmapoint(u,p_cov[k-1],limbda)
    print(sigma1[0][0])
    mean = np.zeros(n)
    beta = 2
    # 1. Update state with IMU inputs

    for i in range((2*n+1)):
        p_check = sigma1[0:3,i]
        print(p_check)
        v_check = sigma1[3:6,i]
        et = sigma1[6:9,i]
        q_check = Quaternion(euler=et).to_numpy()
        q_check = (Quaternion(*q_est[k-1]).normalize()).to_numpy()
        rm = Quaternion(*q_check).to_mat()
        p_check = p_check + delta_t*v_check + 0.5*(delta_t**2)*(rm@imu_f.data[k-1] + g)
    	#v_check = v_check + delta_t*(rm@imu_f.data[k-1] + g)
    	#q_check = Quaternion(axis_angle =delta_t*imu_w.data[k-1]).quat_mult_right(Quaternion(*q_check), out='np')
    	#q_check = (Quaternion(*q_check).normalize()).to_numpy()
    	#eulart = Quaternion(*q_check).to_euler()
    	#if i == 0:
    	#	wn = limbda/(n + limbda)
    	#	wc = wn + 1- alpha**2 + beta

    	#else:
    	#	wn = 1/(2*(n + limbda))
    	#	wc = 1/(2*(n + limbda))

    	#ut = np.zeros([9,1])
    	#ut[0:3,0] = p_check[0:3]
    	#ut[3:6,0] = v_check[0:3]
    	#ut[6:9,0] = eulart
    	#mean = mean + wn*ut
    #p_est[k] = mean[0:3]
    #v_est[k] = mean[3:6]
    #eu = mean[6:9]
    #q_est[k] = Quaternion(euler=eu).to_numpy()
    #q_est[k] = (Quaternion(*q_est[k]).normalize()).to_numpy()

    """
    # 2. Propagate uncertainty
    imuerr = np.eye(6)
    imuerr[0:3,0:3] =(delta_t**2)*(var_imu_f)*np.eye(3)
    imuerr[3:6,3:6] =(delta_t**2)*(var_imu_w)*np.eye(3)
    p_cov[k] = f@p_cov[k-1]@f.T +l_jac@imuerr@l_jac.T
    #print(p_cov[k])
    # 3. Check availability of GNSS and LIDAR measurements

    testl = np.where(lidar.t == imu_f.t[k-1],True,False)
    #print(np.where(gnss.t == imu_f.t[k-1]))
    for i in range(testl.shape[0]):
        if testl[i] == True:
            y_k = lidar.data[i]
            sensor_var = (var_lidar)*np.eye(3)
            p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(sensor_var, p_cov[k], y_k, p_est[k], v_est[k], q_est[k])
    #print(gnss.t)
    #print("___________________")
    #print(imu_f.t[k-1])

    test = np.where(gnss.t == imu_f.t[k-1],True,False)
    #print(np.where(gnss.t == imu_f.t[k-1]))
    for i in range(test.shape[0]):
        if test[i] == True:
            y_k = gnss.data[i]
            sensor_var = (var_gnss)*np.eye(3)
            p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(sensor_var, p_cov[k], y_k, p_est[k], v_est[k], q_est[k])
	"""
    # Update states (save)
#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()

#### 7. Submission #############################################################################

################################################################################################
# Now we can prepare your results for submission to the Coursera platform. Uncomment the
# corresponding lines to prepare a file that will save your position estimates in a format
# that corresponds to what we're expecting on Coursera.
################################################################################################
"""
# Pt. 1 submission
p1_indices = [9000, 9400, 9800, 10200, 10600]
p1_str = ''
for val in p1_indices:
    for i in range(3):
        p1_str += '%.3f ' % (p_est[val, i])
with open('pt1_submission.txt', 'w') as file:
    file.write(p1_str)
"""
 #Pt. 2 submission
p2_indices = [9000, 9400, 9800, 10200, 10600]
p2_str = ''
for val in p2_indices:
    for i in range(3):
        p2_str += '%.3f ' % (p_est[val, i])
with open('pt2_submission.txt', 'w') as file:
    file.write(p2_str)
"""
#Pt. 3 submission
p3_indices = [6800, 7600, 8400, 9200, 10000]
p3_str = ''
for val in p3_indices:
    for i in range(3):
        p3_str += '%.3f ' % (p_est[val, i])
with open('pt3_submission.txt', 'w') as file:
    file.write(p3_str)
"""
