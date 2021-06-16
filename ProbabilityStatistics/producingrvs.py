# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:11:21 2020

@author: CodeTilde
"""
#produing rvs, random samples, ploting pdfs and ploting CDFs 

from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import poisson
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,100)
norm_rv = norm(0,.2)
norm_rv1 = norm(0,1)
norm_rv2 = norm (0,2)

#################################################################
# getting mean, variance, pdf and cdf values
#%%
#"The round() function returns a floating point number that is a rounded version of the specified number, with the specified number of decimals"w3school
print(norm_rv.mean())
print(round(norm_rv.var(),2))
print(norm_rv1.cdf(0))
print(round(norm_rv1.pdf(4),2)) 
#%%

#################################################################
#%%
#plotting cdfs
fig, ax = plt.subplots( 1, 1,  figsize=(10, 6) )
ax.plot(x, norm_rv.cdf(x), 'r-', lw=2, label ="norm(0,.2)")
ax.plot(x, norm_rv1.cdf(x), 'b-', lw=2, label ="norm(0,1)")
ax.plot(x, norm_rv2.cdf(x), 'c-', lw=2, label ="norm(0,2)")
ax.legend()
ax.set_title('CDF plot of Normal distributions')
ax.set_ylabel('cumulative density function')
#fig.savefig("")

#################################################################
#%%
#plotting pdf
fig, ax = plt.subplots( 1, 1,  figsize=(10, 6) )
ax.plot(x, norm_rv.pdf(x), 'r-', lw=2, label ="norm(0,.2)")
ax.plot(x, norm_rv1.pdf(x), 'b-', lw=2, label ="norm(0,1)")
ax.plot(x, norm_rv2.pdf(x), 'c-', lw=2, label ="norm(0,2)")
ax.legend()
plt.show()
ax.set_title('pdf plot of Normal distributions')
ax.set_ylabel('probability density function')
#fig.savefig("")
######################################################
#%%
#producing random samples
norm_sample = norm_rv.rvs(30);
norm_sample1 = norm_rv1.rvs(30);
norm_sample2 = norm_rv2.rvs(30);
fig, axs = plt.subplots(3,1, figsize=(10, 6))
fig.suptitle('Random samples from normal distribution with different variances')
y = [0]*30
axs[0].scatter(norm_sample,y, c='r')#, 'm-')#, label ="norm(0,1)")
axs[0].set_xlim(-3, 3)
axs[1].scatter(norm_sample1,y, c='b')#, label ="norm(0,2)")
axs[1].set_xlim(-3, 3)
axs[2].scatter(norm_sample2,y, c='c')# label ="norm(0,.2)")
axs[2].set_xlim(-3, 3)
#fig.savefig("")
#%%
###########################################################################################
#%%exponential distribution
## pdf = lambda * exp(-lambda * x). 
# scale = 1 / lambda.
expon_rv = expon()
expon_rv1 = expon(loc=0, scale=.5)
expon_rv2 = expon(loc=0, scale=10)

print("scale = 1, mean =?",expon_rv.mean())
print("scale = 1, variance =?",round(expon_rv.var(),2))
print("scale = .5, mean =?", expon_rv1.mean())
print(round(expon_rv1.var(),2))
print(expon_rv1.pdf(0))
print(expon_rv2.mean())
print(round(expon_rv2.var(),2))

x = np.linspace(0,12,100)
#plotting pdf
fig, ax = plt.subplots( 1, 1,  figsize=(10, 6) )
ax.plot(x, expon_rv.pdf(x), 'r-', lw=2, label ="exp(scale=1)")
ax.plot(x, expon_rv1.pdf(x), 'b-', lw=2, label ="exp(scale=.5)")
ax.plot(x, expon_rv2.pdf(x), 'c-', lw=2, label ="exp(scale=10)")
ax.legend()
#fig.savefig("")
################################################################################################
#%%plotting cdf
fig, ax = plt.subplots( 1, 1,  figsize=(10, 6) )
ax.plot(x, expon_rv.cdf(x), 'r-', lw=2, label ="exp(scale=1)")
ax.plot(x, expon_rv1.cdf(x), 'b-', lw=2, label ="exp(scale=.1)")
ax.plot(x, expon_rv2.cdf(x), 'c-', lw=2, label ="exp(scale=10)")
ax.legend()
#fig.savefig("")
###############################################################################
#discrete random variable
#binomial dsitribution
#%%

binomial_rv = stats.binom(10, 0.2)
print("binomial = binom(10, 0.2)")
print ("P(X <= 4) =",round(binomial_rv.cdf(4),2))           # P(X <= 4)
print ("E[X]=np", round(binomial_rv.mean(),2))          # E[X]
print ("Var = npq= ",round(binomial_rv.var(),2))            # var(X)
print ("std = ",round(binomial_rv.std(),2))            # std(X)
print ("A random sample", binomial_rv.rvs())            # A random sample from X
print ("10 random samples", binomial_rv.rvs(10))
print("pmf at n=9", round(binomial_rv.pmf(10),5))
k =range(11);  
fig, ax = plt.subplots( 1, 1,  figsize=(10, 6) )
ax.bar(k, binomial_rv.pmf(k),align='center', alpha=1)
ax.set_ylabel("probability mass function")
#fig.savefig("")

#########################################################################%
#discrete random variable
#poisson distribution

poisson_rv = poisson(0.5)
print("poisson = poisson(0.5)")
print ("P(X <= 4) =",round(poisson_rv.cdf(4),2))           # P(X <= 4)
print ("E[X]=np", round(poisson_rv.mean(),2))          # E[X]
print ("Var = npq= ",round(poisson_rv.var(),2))            # var(X)
print ("std = ",round(poisson_rv.std(),2))            # std(X)
print ("A random sample", poisson_rv.rvs())            # A random sample from X
print ("10 random samples", poisson_rv.rvs(10))
print("pmf at n=10", round(poisson_rv.pmf(4),5))
k =range(10);  
fig, ax = plt.subplots( 1, 1,  figsize=(10, 6) )
ax.bar(k, poisson_rv.pmf(k),align='center', alpha=1)
ax.set_ylabel("probability mass function")
ax.set_title("pmf plot of poisson distribution")
#fig.savefig("")
