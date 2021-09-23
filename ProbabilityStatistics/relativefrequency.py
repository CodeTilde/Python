#relative frequency interpretation simulation 
import random
import matplotlib.pyplot as myplt

def coin_toss ():
    if random.random() <= 0.5:
        print("HEAD")
    else:
        print("TAIL")        
    
def coin_trial( trial_num ):
    heads = 0
    for i in range(trial_num):
        if random.random() <= 0.5:
            heads +=1
    return heads/trial_num

number_list =[ 100, 200, 300,400, 500,600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
prob_list = []

for num in number_list:
    prob_list.append(coin_trial(num))

fig, ax = myplt.subplots( 1, 1,  figsize=(10, 6) )
ax.plot(number_list,  prob_list,)
ax.set_xlabel('number of trails')
ax.set_ylabel('probability')
ax.set_title('Relative frequency interpretation')
#fig.savefig("")
  
