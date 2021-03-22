from kalman import kalman_filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

""" setting up variables """
mu_x = 5  # mean of X (true value)
mu_y = 8  # mean of Y (true value)
sigma_x = 2  # standard deviation of X
sigma_y = 0.2  # standard deviation of Y
max_samples = 50  # number of samples displayed before scrolling

R = np.array([
    [sigma_x, 0],
    [0, sigma_y]
])
H = np.eye(2)
kf = kalman_filter(R, H)

x = np.array([[0, 0]]).T  # initial state estimate
p = np.array([[1000, 0], [0, 1000]])  # initial state uncertainty

""" Plotting """
fig1, (ax1, ax2) = plt.subplots(2, 1)

# true value line
ax1.axhline(mu_x, color='r', label='True Value')
ax2.axhline(mu_y, color='r', label='True Value')

# measurement line
line_x_measured, = ax1.plot([-1], [0], '+k', label='X Measured')
line_y_measured, = ax2.plot([-1], [0], '+k', label='Y Measured')

# filtered value line
line_x_filtered, = ax1.plot([-1], [0], '--b', label='X Filtered')
line_y_filtered, = ax2.plot([-1], [0], '--b', label='Y Filtered')

# display legend
ax1.legend(loc='lower left')
ax2.legend(loc='lower left')

def animator(i):
    global x, p, zx, zy
    im = i % 2
    io = i // 2

    def update_limits(ax, line):
        xdata = line.get_xdata()
        if xdata.size >= max_samples:
            ax.set_xlim(left=xdata[-max_samples], right=xdata[-1] + 2)
        else:
            ax.set_xlim(left=xdata[0], right=xdata[-1] + 2)
        return

    def update_line(line, x, y):
        xdata = np.hstack((line.get_xdata()[-max_samples:], x))
        ydata = np.hstack((line.get_ydata()[-max_samples:], y))

        line.set_xdata(xdata)
        line.set_ydata(ydata)

        return

    if im == 0:
        # generate noisy measurement
        zx = np.random.normal(mu_x, sigma_x)
        zy = np.random.normal(mu_y, sigma_y)

        # plot the measurement
        update_limits(ax1, line_x_measured)
        update_limits(ax2, line_y_measured)
        ax1.set_ylim(bottom=-3.1 * sigma_x + mu_x, top=3.1 * sigma_x + mu_x)
        ax2.set_ylim(bottom=-3.1 * sigma_y + mu_y, top=3.1 * sigma_y + mu_y)
        update_line(line_x_measured, io, zx)
        update_line(line_y_measured, io, zy)
        
    else:
        # filter the measurement
        x, p = kf.update(x, p, np.vstack((zx, zy)))

        # plot the filtered value
        update_line(line_x_filtered, io, x[0])
        update_line(line_y_filtered, io, x[1])

    return line_x_measured, line_y_measured, 

# start animation
ani = animation.FuncAnimation(fig1, animator, interval=1)
plt.show(block=True)
