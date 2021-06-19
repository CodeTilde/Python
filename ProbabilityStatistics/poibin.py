#Poisson approximates binomial distribution
from scipy.stats import binom
from scipy.stats import poisson
import matplotlib.pyplot as plt
def binom_gen(n,p):
    binom_rv = binom(n, p)
    lam = n*p
 #   sigma= math.sqrt(n*p*(1-p))
    k = range(int(5*lam))
    binom_pmf =   binom_rv.pmf(k)        # binom.pmf(k1, n, .1)
    return k, binom_pmf
#######################################################################################
#%%
n_list=[20,  1000, 1500, 2000]
p_list=[.1, .5, .01, .01]
fig, ax = plt.subplots( 4, 1,  figsize=(10, 7) )
count=0;
size = len(n_list)
for count in range(size):
    n = n_list[count]
    p = p_list[count]
    k, binom_pmf = binom_gen(n,p)
    poisson_rv = poisson(n*p)
    ax[count].bar(k, binom_pmf,color= 'r', label ='Binomial' )
    ax[count].bar(k,poisson_rv.pmf(k),color='y', width=.4, label='Poisson')
    ax[count].legend()
    ax[count].set_ylabel('pmf')
    count = count+1
    
fig.suptitle('Poisson approximates binomial distribution')
#fig.savefig("")
