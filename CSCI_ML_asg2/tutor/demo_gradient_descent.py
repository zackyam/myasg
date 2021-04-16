from tu3utils import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
import time
from matplotlib import rc
import numpy as np

plt.style.use('ggplot')


def show_data(points):
    x = points[:, 0]
    y = points[:, 1]

    ax = plt.gca()
    f1, = plt.plot(x, y, 'x', label="Training Data", MarkerSize=10)
    legend = plt.legend(frameon=1,
                        handler_map={f1: HandlerLine2D(numpoints=1)})
    frame = legend.get_frame()
    frame.set_color('white')
    plt.xlabel(r'Normalized Horse Power', fontsize=20)
    plt.ylabel(r'Price (K\$)', fontsize=20)
    plt.grid(True)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ymajor_ticks = np.arange(0, 51, 5)
    xmajor_ticks = np.arange(-2, 7, 2)

    ax.set_xticks(xmajor_ticks)
    ax.set_yticks(ymajor_ticks)

    ax.grid(True)
    plt.savefig('train_data', bbox_inches='tight', pad_inches=0)


def demo_figure(it, theta0_track, theta1_track, error_track, points):
    x = points[:, 0]
    y = points[:, 1]

    t0 = linspace(0, 2 * theta0_track[-1], 50)
    t1 = linspace(0, 2 * theta1_track[-1], 50)
    e = zeros((len(t0), len(t1)))
    for i in range(len(t0)):
        for j in range(len(t1)):
            e[i, j] = error_function(t0[i], t1[j], points)
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    f1, = plt.plot(x, y, 'x', label="Training Data", MarkerSize=10)

    legend = plt.legend(frameon=1,
                        handler_map={f1: HandlerLine2D(numpoints=1)})
    frame = legend.get_frame()
    frame.set_color('white')
    plt.xlabel(r'Normalized Horse Power', fontsize=20)
    plt.ylabel(r'Price (K\$)', fontsize=20)
    plt.grid(True)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ymajor_ticks = np.arange(0, 51, 5)
    xmajor_ticks = np.arange(-2, 7, 2)

    ax.set_xticks(xmajor_ticks)
    ax.set_yticks(ymajor_ticks)

    ax.grid(True)

    plt.subplot(1, 2, 2)

    ax = plt.gca()
    CT = plt.contourf(t0, t1, e, logspace(0, 2, 30), cmap=plt.cm.jet)
    CT2 = plt.contour(t0, t1, e, logspace(0, 2, 30), cmap=plt.cm.jet)
    plt.xlabel(r'${\theta}_0$', fontsize=20)
    plt.ylabel(r'${\theta}_1$', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.colorbar(CT, aspect=10)
    for coll in CT.collections:
        coll.remove()
    current_it0 = theta0_track[it]
    current_it1 = theta1_track[it]
    current_err = error_track[it]
    plt.subplot(1, 2, 1)
    f2, = plt.plot(x,
                   current_it0 + current_it1 * x,
                   '-',
                   MarkerSize=5,
                   linewidth=3,
                   label="Current Regression Line")
    legend = plt.legend(frameon=1,
                        handler_map={f1: HandlerLine2D(numpoints=1)})
    frame = legend.get_frame()
    frame.set_color('white')
    plt.subplot(1, 2, 2)
    plt.suptitle("Gradient Descent Demo: iteration " + str(it), fontsize=30)
    plt.plot(current_it0, current_it1, 'b+', MarkerSize=10, linewidth=3)

    figname = 'demo' + str(it)
    print ("it:", it, "theta0:", current_it0, "theta1:", current_it1, "error1:", current_err)
    plt.savefig(figname, bbox_inches='tight', pad_inches=0)


def runner(learning_rate=0.01, initial_theta_0=0, initial_theta_1=0, num_iterations=1000):
    points = genfromtxt('imports-85.data', delimiter=',', usecols=(21, 25))
    points = points[~isnan(points).any(axis=1)]
    points[:, 1] = points[:, 1] / 1000
    # normalize
    points[:, 0] = normalize_feature(points[:, 0])

    print ("Starting gradient descent at theta_0 = {0}, theta_1 = {1}, error = {2}".format(
        initial_theta_0, initial_theta_1, error_function(
            initial_theta_0, initial_theta_1, points)))
    print ("Running...")
    [theta_0, theta_1, theta_0_track, theta_1_track, error_track
     ] = gradient_descent_method(points, initial_theta_0, initial_theta_1,
                                 learning_rate, num_iterations)
    print ("After {0} iterations theta_0 = {1}, theta_1 = {2}, error = {3}".format(
        num_iterations, theta_0, theta_1, error_function(theta_0, theta_1,
                                                         points)))
    demo_figure(0, theta_0_track, theta_1_track, error_track, points)
    # show_data(points)
    for it in range(num_iterations):
        if it % 10 == 0:
            demo_figure(it, theta_0_track, theta_1_track, error_track, points)


runner()
