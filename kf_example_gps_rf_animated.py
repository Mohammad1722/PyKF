from kalman import kalman_filter
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation


rnd_gen = np.random.default_rng()  # random number rnd_generator
# rnd_gen = np.random.default_rng(seed=951753)

''' Setup variables '''
mu_pos = np.array([[10], [10]])  # True position of vehicle x,y
ref_obj = np.array([[0], [0]])  # True position of the reference object detected by the rangefinder
mu_range = np.linalg.norm(ref_obj - mu_pos)  # true range
theta_rf = np.arctan2((ref_obj - mu_pos)[1], (ref_obj - mu_pos)[0])[0]  # angle of rangefinder with global x axis (assuming that the reference object is at the origin)
cov_gps = np.array([[5, 0], [0, 5]])  # covariance of the GPS
sigma2_rf = 1  # variance of the Rangefinder


H_gps = np.eye(2)
H_rf = np.eye(2)
H_rf2 = np.array([[cos(theta_rf)], [sin(theta_rf)]])

R_gps = cov_gps
R_rf = H_rf2 @ np.array([[sigma2_rf]]) @ H_rf2.T  # covariance of rf

x = np.array([[0], [0]])  # initial state estimate
p = np.array([[1e6, 0], [0, 1e6]])  # initial state uncertainty

''' Prepare the plot '''
fig1, ax1 = plt.subplots()
ax1.set_aspect(1)
ax1.axhline(0, color='k')
ax1.axvline(0, color='k')

ax1.add_patch(plt.Circle((mu_pos[0], mu_pos[1]), 0.5, color='g', fill=False, label='True Position'))

line_gps, = ax1.plot(x[0,0], x[1,0], 'xb', markersize=10, label='GPS measurement')
line_rf, = ax1.plot(x[0,0], x[1,0], 'xc', markersize=10, label='RF measurement')
circle_est_pos = ax1.add_patch(plt.Circle((x[0,0], x[1,0]), 0.5, color='r', fill=False, label='Estimated Position', ls='-.'))

ax1.legend(loc='lower right')
def animator(i):
    global x, p
    
    def update_line(line, x):
        line.set_xdata(x[0, 0])
        line.set_ydata(x[1, 0])
        return

    # generate noisy measurement
    z_gps = rnd_gen.multivariate_normal(mu_pos.flatten(), R_gps).reshape(2, 1)
    z_rf = np.array([[rnd_gen.normal(mu_range, sigma2_rf)]])
    z_rf = ref_obj - H_rf2 @ z_rf

    # apply kalman filter
    x, p = kalman_filter.filter(x, p, z_gps, R_gps, H_gps)
    x, p = kalman_filter.filter(x, p, z_rf, R_rf, H_rf)
    
    # update plot
    update_line(line_gps, z_gps)
    update_line(line_rf, z_rf)
    circle_est_pos.center = (x[0,0], x[1,0])

    ax1.set_xlim(left=np.min([ax1.get_xlim()[0], z_rf[0,0], z_gps[0,0], x[0,0], mu_pos[0,0]]),
                right=np.max([ax1.get_xlim()[1], z_rf[0,0], z_gps[0,0], x[0,0], mu_pos[0,0]]))
    
    ax1.set_ylim(bottom=np.min([ax1.get_ylim()[0], z_rf[1,0], z_gps[1,0], x[1,0], mu_pos[1,0]]),
                top=np.max([ax1.get_ylim()[1], z_rf[1,0], z_gps[1,0], x[1,0], mu_pos[1,0]]))

    return line_gps, line_rf, circle_est_pos, 

# start animation
ani = animation.FuncAnimation(fig1, animator, interval=500)
plt.show(block=True)

