from kalman import kalman_filter
import numpy as np
import matplotlib.pyplot as plt

ns = 50  # number of samples
mu_x = 5  # mean of X (true value)
mu_y = 0  # mean of Y (true value)
sigma_x = 0.3  # standard deviation of X
sigma_y = 0.2  # standard deviation of Y

x_measured = np.random.normal(mu_x, sigma_x, ns)  # simulate noisy sensor measurements for X
y_measured = np.random.normal(mu_y, sigma_y, ns)  # simulate noisy sensor measurements for Y
measurements = np.vstack((x_measured, y_measured))

R = np.array([
    [sigma_x, 0],
    [0, sigma_y]
])
H = np.eye(2)
kf = kalman_filter(R, H)


x = np.array([[0, 0]]).T  # initial state estimate
p = np.array([[1000, 0], [0, 1000]])  # initial state uncertainty
xf = x.copy()  # array to store all estimated states (filtered values)

# pass measurements to filter
for i in range(measurements.size):
    z = measurements[:, i:i+1]
    x, p = kf.update(x, p, z)
    xf = np.hstack((xf, x))

''' THEN USE FILTERED VALUES xf '''
plt.axhline(mu_x, label='True Value', color='r')
plt.plot(measurements[0, :], 'b', label='X measured')
plt.plot(np.arange(0, ns), xf[0, 1:], 'g', label='X filtered')
plt.legend()
plt.show()

plt.axhline(mu_y, label='True Value', color='r')
plt.plot(measurements[1, :], 'b', label='Y measured')
plt.plot(np.arange(0, ns), xf[1, 1:], 'g', label='Y filtered')
plt.legend()
plt.show()

