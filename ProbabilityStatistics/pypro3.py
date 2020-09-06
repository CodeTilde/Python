# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:09:23 2020

central limit theorem 
smaple mean X (x1+x2+........xn)/n = norm(mean, sigma)
xi : identical independent distributed rvs normal uniform binom poisson binom bernouli.. 
sample_mean_list produces the smaple mean : X = X1+X2+.....Xn/n
produces the sample means for differen K listed in list_len
produced  sample_len  samples  of sample mean
plot histogram of sample mean to see how sample mean approaches a Normal distribution 
@author: Najmeh

"""
from scipy.stats import binom
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import uniform
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import bernoulli
#from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import math
def sample_mean_list(random_var, n, list_len):
    sample_list = []
    for i in range (list_len):
        sample_list.append(sum( random_var.rvs (n))/n)
    return  sample_list   
    
def plot_dist(n_list, sample_len, dist, dist_name):
    samples=[]
    fig, axs = plt.subplots(4, 1)
    for i in range(len(n_list)):
        samples= sample_mean_list(dist, n_list[i], sample_len )
        axs[i].hist(samples,bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
    plt.savefig("cumulative_density_distribution_04.jpg", bbox_inches='tight')
    plt.show()
    plt.close

    
mean = 0
sigma = 2
norm_rv = norm(mean, sigma) 
a = 2
b = 6
uniform_rv = uniform(a,b)
n = 100
p = .1
binom_rv = binom(n,p) 
  
lambd = 4 
expon_rv = expon(lambd)

lambd =2
poisson_rv = poisson(lambd)

p = .4
berno_rv = bernoulli(p)


n_list = [1, 5,  50, 100, 100000]

sample_len = 1000
plot_dist(n_list, sample_len ,expon_rv,'normal')