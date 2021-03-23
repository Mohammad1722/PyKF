from kalman import kalman_filter
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

gen = np.random.default_rng()
# gen = np.random.default_rng(seed=951753)

''' Setup variables '''
ns = 1000  # number of samples

mu_pos = 20  # True position 
sigma2_s1 = 10  # variance of first sensor (very noisy)
sigma2_s2 = 10  # variance of second sensor (very noisy)

s1_readings = gen.normal(mu_pos, sigma2_s1, ns)  # simulate noisy sensor measurements for s1
s2_readings = gen.normal(mu_pos, sigma2_s2, ns)  # simulate noisy sensor measurements for s2

R1 = np.array([[sigma2_s1]])
R2 = np.array([[sigma2_s2]])

H1 = np.eye(1)
H2 = np.eye(1)

x = np.array([[0]])  # initial state estimate
p = np.array([[1e6]])  # initial state uncertainty
xf = x.copy()  # array to store all estimated states (filtered values)

''' Apply KF '''
for i in range(ns):
    z1 = s1_readings[i]
    z2 = s2_readings[i]
    x, p = kalman_filter.filter(x, p, z1, R1, H1)
    x, p = kalman_filter.filter(x, p, z2, R2, H2)
    xf = np.hstack((xf, x))

''' USE FILTERED VALUES xf '''
plt.axhline(mu_pos, label='True Value', color='r')
plt.plot(s1_readings, 'b--', label='sensor 1')
plt.plot(s2_readings, 'c--', label='sensor 2')
plt.plot(np.arange(0, ns), xf[0, 1:], 'g', label='filtered')
plt.legend()
plt.show()

plt.axhline(mu_pos, label='True Value', color='r')
plt.plot(s1_readings, 'b+', label='sensor 1')
plt.plot(s2_readings, 'cx', label='sensor 2')
plt.plot(np.arange(0, ns), xf[0, 1:], 'g', label='filtered')
plt.legend()
plt.show()

