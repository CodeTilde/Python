#Normal approximation of binomial distrubution  
#central limit theorem 
from scipy.stats import expon
from scipy.stats import uniform
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
###########################################################################
def binom_gen(n,p):
    binom_rv = binom(n, p)
    mean = n*p
    sigma= math.sqrt(n*p*(1-p))
    k = range(int(mean-3*sigma), int(mean+3*sigma),1)
    P_binom =   binom_rv.pmf(k)        # binom.pmf(k1, n, .1)
    return  mean, sigma, k, P_binom

n_list =[ 10, 100, 1000,10000]
prob_list = []
r =2;
col =2;
n = 10
p=.1
fig = plt.figure(figsize=(15,15))
lenn = len(n_list)
r = lenn/2
r = 0
col = 0
i =1;
for n in n_list :
    mean, sigma, k, binom_rv = binom_gen(n,p)
    norm_rv = norm(mean, sigma)
    ax = fig.add_subplot(2,2,i)
    ax.plot(k, binom_rv,color ='r',label='binomial')
    ax.plot(k, norm_rv.pdf(k),color='b', label='normal')
    ax.set_xlabel('$X$', fontsize=13)
    ax.set_ylabel('$probability$', fontsize=13)
    ax.set_title('p=.1 n='+str(n), fontsize=15)
    ax.legend()
    i = i+1
fig.suptitle('Approximation of a binomial by  Normal dsitribution',fontsize=25)    
#fig.savefig("")
####################################################################################################
#central limit theorem
def sample_mean_list(random_var, s_size, n):
    sample_list = [] 
    for i in range (n):
        row = random_var.rvs(s_size)
        sample_list.append(sum(row)/s_size)        
    return  sample_list   
    
def plot_dist(dist, sample_size, sample_num):
    i=0;
    fig = plt.figure(figsize=(24,6))
    list_len = len(sample_size)
    i=1
    for s_size in sample_size:
        hist_samples = sample_mean_list(dist, s_size, sample_num)
        ax = fig.add_subplot(1,list_len,i)
        ax.hist(hist_samples,10, density=True)
        #sns.distplot(hist_samples)
        plt.grid(axis='y', alpha=0.75)
        #ax.set_ylabel('$histogram$', fontsize=13)
        ax.set_title('n='+str(s_size), fontsize=15)
        i=i+1
    return fig

    
mean = 0
sigma = 2
norm_rv = norm(mean, sigma) 
a = 2
b = 6
uniform_rv = uniform(a,b)
n = 1000
p = .1
binom_rv = binom(n,p) 
  
lambd = 4 
expon_rv = expon(lambd)

lambd =2
poisson_rv = poisson(lambd)
#####################################################################################################
sample_size = [1, 5,  20, 40, 100]

sample_num = n;

dist = norm_rv
fig = plot_dist(dist, sample_size, sample_num)
fig.suptitle('Central limit theorem applied to normal dsitribution',fontsize=25)    
#fig.savefig("")

dist = uniform_rv
fig = plot_dist(dist, sample_size, sample_num)
fig.suptitle('Central limit theorem applied to uniform dsitribution',fontsize=25)    
#fig.savefig("")


dist = binom_rv
fig = plot_dist(dist, sample_size, sample_num)
fig.suptitle('Central limit theorem applied to binomial dsitribution',fontsize=25)    
#fig.savefig(" ")

dist = expon_rv
fig = plot_dist(dist, sample_size, sample_num)
fig.suptitle('Central limit theorem applied to exponential dsitribution',fontsize=25)    
#fig.savefig("")


dist = poisson_rv
fig = plot_dist(dist, sample_size, sample_num)
fig.suptitle('Central limit theorem applied to poisson dsitribution',fontsize=25)    
#fig.savefig("")


