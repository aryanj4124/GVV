import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

#Number of samples is 6, 1 denotes heads and 0 denotes tails
simlen=int(6)

#Probability that heads occurred, i.e., X=1 is 1/2
prob = 0.5

#Generating sample date using Bernoulli r.v.
data_bern = bernoulli.rvs(size=simlen,p=prob)

#Calculating the number of heads
arr_heads = np.nonzero(data_bern == 1)
num_heads = np.size(arr_heads)

#calculating the number of tails
num_tails = 6 - num_heads


print("samples generated:",data_bern)
print("number of heads:",num_heads)
print("number of tails:",num_tails)
print("X:",num_heads - num_tails)
