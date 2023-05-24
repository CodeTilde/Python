# a cheat sheet for basic  ML algorthms.
 #Most of them are from 
#https://github.com/epfml/ML_course/blob/master/labs
# I will add some new resources



#Write a function that accepts data matrix x ∈ R n×d as input and outputs the same data after normalization. n
#is the number of samples, and d the number of dimensions, i.e. rows contain samples and columns features.
#pip install numpy, sklearn
import numpy as np

#from sklearn import preprocessing
#######################################################################################################
#Standardization/Normalization:
def standardize(x):
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return std_data
#from sklearn import preprocessing
#preprocessing.normalize(x, axis =0)
#Standardize features by removing the mean and scaling to unit variance.
#The standard score of a sample x is calculated as:
# z = (x - u) / s
# preprocessing.StandardScaler()

##############################################################################################################
#Write a function that accepts two matrices P ∈ R p×2 , Q ∈ R q×2 as input, where each row contains the (x, y)
#coordinates of an interest point. Note that the number of points (p and q) do not have to be equal. As output,
#compute the pairwise distances of all points in P to all points in Q and collect them in matrix D. Element D i,j
#is the Euclidean distance of the i-th point in P to the j-th point in Q.

from scipy.spatial.distance import cdist
def distance (a, b):
	a_size= a.shape
	b_size= b.shape
	print (a_size)
	print (b_size)
	distance_mat = np.zeros([a_size[0],b_size[0]])
	#a_size[1] = b_size[1]
	
	for i in range (a_size[0]):
		arr = a[i]
		temp = np.vstack([arr]*b_size[0])
		row = np.sqrt (np.power((temp - b),2).sum(axis=1))
		#print(row)
		distance_mat[i]= row 	
	return distance_mat


def with_indices(p, q):
    rows, cols = np.indices((p.shape[0], q.shape[0]))
    distances = np.sqrt(np.sum((p[rows.ravel(), :] - q[cols.ravel(), :])**2, axis=1)) # ravel = .reshape(-1)
    return distances.reshape((p.shape[0], q.shape[0]))

def scipy_version(p, q):
    return cdist(p, q)
    
def tensor_broadcasting(p, q):
    return np.sqrt(np.sum((p[:,np.newaxis,:]-q[np.newaxis,:,:])**2, axis=2))    
#########################################################################################

#As input, your function receives
#a set of data examples x n ∈ R d (indexed by 1 ≤ n ≤ N ) as well as the two sets of parameters θ 1 = (µ 1 , Σ 1 )
#and θ 2 = (µ 2 , Σ 2 ) of two given multivariate Gaussian distributions:
#x_data n*D teta1 = teta2

# Data generation for likelihood_classifier
#from numpy.random import rand, randn
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, randn

n, d, k = 100, 2, 2

np.random.seed(20)
X = rand(n, d)
means = [rand(d) * 0.5 + 0.5 , - rand(d)  * 0.5 + 0.5]  
S = np.diag(rand(d))
sigmas = [S]*k 


def compute_log_p(X, mean, sigma):
    d = X.shape[1]
    dxm = X - mean
    exponent = -0.5 * np.sum(dxm * np.dot(dxm, np.linalg.inv(sigma)), axis=1)
    return exponent - np.log(2 * np.pi) * (d / 2) - 0.5 * np.log(np.linalg.det(sigma))

log_ps = [compute_log_p(X, m, s) for m, s in zip(means, sigmas)]  # exercise: try to do this without looping
assignments = np.argmax(log_ps, axis=0)
print(assignments)


colors = np.array(['red', 'green'])[assignments]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=100)
plt.scatter(np.array(means)[:, 0], np.array(means)[:, 1], marker='*', s=200)
plt.show()



from scipy.stats import multivariate_normal	
def likelihood_classifier (x_data, teta1, teta2):
	normal1 = multivariate_normal(mean=teta1[0], cov=teta1[1])
	normal2 = multivariate_normal(mean=teta2[0], cov=teta2[1])
	assignments = normal1.pdf(x_data) > normal2.pdf(x_data)
	return assignments
	

assignments = 	likelihood_classifier (X, means, sigmas) #does the same as above 		
			


