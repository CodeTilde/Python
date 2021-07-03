import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math
from scipy.stats import norm
from matplotlib import cm
from numpy import linalg as LA
def func1(mean,covariance, grid_size):
    x = np.linspace(-6, 6, grid_size)
    y = np.linspace(-6, 6, grid_size)
    x_prim,y_prim = np.meshgrid(x, y)
    rv = multivariate_normal(mean, covariance)
    point_set=np.dstack((x_prim, y_prim))
    return x_prim, y_prim, point_set, rv

mean=[0 ,0]
cov= np.array([[1 ,0],[0, 1]])
grid_size =100
x,y, points, rv = func1(mean, cov, grid_size)
random_points = rv.rvs(50)
#%%
##producing a normal distribution
fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(1,2,1, projection='3d')
surf = ax.plot_surface(x, y, rv.pdf(points), cmap=cm.YlGnBu)
ax.set_xlabel('$X$', fontsize=15)
ax.set_ylabel('$Y$', fontsize=15)
ax.set_title('Bivariate normal pdf', fontsize=15)

ax = fig.add_subplot(1,2,2)
con = ax.contourf(x, y,rv.pdf(points) , 30, cmap=cm.YlGnBu)
ax.scatter(random_points[:,0], random_points[:,1],color='r',alpha=.6)
ax.set_xlabel('$X$', fontsize=15)
ax.set_ylabel('$Y$', fontsize=15)
ax.axis([-3, 3, -3, 3])
ax.set_aspect('equal')
ax.set_title('Level sets and Samples', fontsize=15)
cbar = plt.colorbar(con)
cbar.ax.set_ylabel('density: $p(y_1, y_2)$', fontsize=13)
fig.suptitle('Bivariate Normal',fontsize=25)
#fig.savefig("")
#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots( 2, 2,  figsize=(12, 12) )
mean=[0 ,0]
cov= np.array([[3 ,0],[0, 1]])
x,y, points0, rv0 = func1(mean, cov, grid_size)
random_points0 = rv0.rvs(50)
#ax = fig.add_subplot(2,2,1)
con = ax0.contourf(x, y,rv0.pdf(points0) , 30, cmap=cm.YlGnBu)
ax0.scatter(random_points0[:,0], random_points0[:,1],color='r',alpha=.6)
ax0.set_xlabel('$X$', fontsize=15)
ax0.set_ylabel('$Y$', fontsize=15)
ax2.axis([-6, 6, -6, 6])
ax2.set_aspect('equal')

mean=[0 ,0]
cov= np.array([[1 ,0],[0, 3]])
x,y, points1, rv1 = func1(mean, cov, grid_size)
random_points1 = rv1.rvs(50)
#ax = fig.add_subplot(2,2,2)
con = ax1.contourf(x, y,rv1.pdf(points1) , 30, cmap=cm.YlGnBu)
ax1.scatter(random_points1[:,0], random_points1[:,1],color='r',alpha=.6)
ax1.set_xlabel('$X$', fontsize=15)
ax1.set_ylabel('$Y$', fontsize=15)
ax2.axis([-6, 6, -6, 6])
ax2.set_aspect('equal')

mean=[0 ,0]
cov= np.array([[1 ,.8],[.8, 1]])
x,y, points2, rv2 = func1(mean, cov, grid_size)
w, v = LA.eig(cov)
random_points2 = rv2.rvs(50)

con = ax2.contourf(x, y,rv2.pdf(points2) , 30, cmap=cm.YlGnBu)
ax2.scatter(random_points2[:,0], random_points2[:,1],color='r',alpha=.6)
ax2.plot([-6*v[0][0], 6*v[0][0]], [-6*v[1][0],6*v[1][0]], 'k--')
ax2.plot([-6*v[0][1], 6*v[0][1]], [-6*v[1][1],6*v[1][1]], 'm--')
ax2.set_xlabel('$X$', fontsize=15)
ax2.set_ylabel('$Y$', fontsize=15)
ax2.axis([-6, 6, -6, 6])
ax2.set_aspect('equal')

mean=[2 ,2]
cov= np.array([[1 ,0.8],[0.8, 1]])
x,y, points3, rv3 = func1(mean, cov, grid_size)
random_points3 = rv3.rvs(50)

con = ax3.contourf(x, y,rv3.pdf(points3) , 30, cmap=cm.YlGnBu)
ax3.scatter(random_points3[:,0], random_points3[:,1],color='r',alpha=.6)
ax3.axis([-6, 6, -6, 6])
ax3.set_aspect('equal')
ax3.set_xlabel('$X$', fontsize=15)
ax3.set_ylabel('$Y$', fontsize=15)

fig.suptitle('Bivariate Normal:Different mean and covariance matrices',fontsize=25)
#fig.savefig("")
#%%













