#P(X>=Y) using relative frequency interpretation
#p(X>=Y), theory lambday/(lambdax+lambday)
from scipy.stats import expon
import matplotlib.pyplot as plt
def prob_cal(n):   
    rv_x = expon(loc=0,scale = 1)
    rv_y = expon(loc=0,scale = 5)
    x_samples = rv_x.rvs(n)
    y_samples = rv_y.rvs(n)
    number = 0
    for i,j in zip(x_samples, y_samples):
        z = i>j
        if (z):
            number = number+1        
    return number/n
number_list =[ 100, 200, 300,400, 500,600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000,6000]
prob_list=[]
for num in number_list:
    prob_list.append(prob_cal(num))

fig, ax = plt.subplots( 1, 1,  figsize=(10, 6) )
ax.plot(number_list,  prob_list,)
ax.set_xlabel('number of trails')
ax.set_ylabel('probability')
ax.set_title('probability of p(X>=Y)')
#fig.savefig("")
