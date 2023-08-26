import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import math

#6 Coins are tossed, 1 denotes heads and 0 denotes tails. There are 1000 samples
simlen= 100000

#Probability that heads occurred, i.e., X=1 is 1/2
prob = 0.5

#Generating sample date using Bernoulli r.v.
p1 = bernoulli.rvs(size=simlen,p=prob)
p2 = bernoulli.rvs(size=simlen,p=prob)
p3 = bernoulli.rvs(size=simlen,p=prob)
p4 = bernoulli.rvs(size=simlen,p=prob)
p5 = bernoulli.rvs(size=simlen,p=prob)
p6 = bernoulli.rvs(size=simlen,p=prob)
#Calculating the number of heads
num_heads = p1 + p2 + p3 + p4 + p5 + p6

#calculating the number of tails
num_tails = 6 - num_heads

X = num_heads - num_tails


print("Probability through simulation")
print("X: -6 =", np.count_nonzero(X==-6)/simlen)
print("X: -4 = ", np.count_nonzero(X==-4)/simlen)
print("X: -2 =", np.count_nonzero(X==-2)/simlen)
print("X: 0 =", np.count_nonzero(X==0)/simlen)
print("X: 2 =", np.count_nonzero(X==2)/simlen)
print("X: 4 =", np.count_nonzero(X==4)/simlen)
print("X: 6 =", np.count_nonzero(X==6)/simlen)

print("Theoretical Probability")
print("X: -6 =", math.comb(6,0)/2**6)
print("X: -4 =", math.comb(6,1)/2**6)
print("X: -2 =", math.comb(6,2)/2**6)
print("X: 0 =", math.comb(6,3)/2**6)
print("X: 2 =", math.comb(6,4)/2**6)
print("X: 4 =", math.comb(6,5)/2**6)
print("X: 6 =", math.comb(6,6)/2**6)



