import random
import math
import numpy as np
import matplotlib.pyplot as plt

def simulate_four_kings():
    deck = [1] * 4 + [0] * 48  # 1 represents a king, 0 represents a non-king
    random.shuffle(deck)
    drawn_cards = deck[:4]
    if (sum(drawn_cards) == 4):
       return True

def simulate_three_kings():
    deck = [1] * 4 + [0] * 48  # 1 represents a king, 0 represents a non-king
    random.shuffle(deck)
    drawn_cards = deck[:4]
    if (sum(drawn_cards) == 3):
       return True
def simulate_two_kings():
    deck = [1] * 4 + [0] * 48  # 1 represents a king, 0 represents a non-king
    random.shuffle(deck)
    drawn_cards = deck[:4]
    if (sum(drawn_cards) == 2):
       return True

def simulate_one_kings():
    deck = [1] * 4 + [0] * 48  # 1 represents a king, 0 represents a non-king
    random.shuffle(deck)
    drawn_cards = deck[:4]
    if (sum(drawn_cards) == 1):
       return True

def simulate_zero_kings():
    deck = [1] * 4 + [0] * 48  # 1 represents a king, 0 represents a non-king
    random.shuffle(deck)
    drawn_cards = deck[:4]
    if (sum(drawn_cards) == 0):
       return True


num_simulations = 1000000  # Number of simulations to run
count1 = 0 
count2 = 0
count3 = 0
count4 = 0
count5 = 0 

for _ in range(num_simulations):
    if simulate_zero_kings():
    	count1 = count1 + 1
    elif simulate_one_kings():
    	count2 = count2 + 1
    elif simulate_two_kings():
    	count3 = count3 + 1
    elif simulate_three_kings():
    	count4 = count4 + 1
    elif simulate_four_kings():
    	count5 = count5 + 1
        
prob1 = count1/num_simulations
prob2 = count2/num_simulations
prob3 = count3/num_simulations
prob4 = count4/num_simulations
prob5 = count5/num_simulations

print("Simulated Probability:")
print("probability 0 kings shown up =",prob1)
print("probability 1 kings shown up =",prob2)
print("probability 2 kings shown up =",prob3)
print("probability 3 kings shown up =",prob4)
print("probability 4 kings shown up =",prob5)


print("Actual Probability:")
print("probability 0 kings shown up =",math.comb(4,0)*math.comb(48,4)/math.comb(52,4))
print("probability 1 kings shown up =",math.comb(4,1)*math.comb(48,3)/math.comb(52,4))
print("probability 2 kings shown up =",math.comb(4,2)*math.comb(48,2)/math.comb(52,4))
print("probability 3 kings shown up =",math.comb(4,3)*math.comb(48,1)/math.comb(52,4))
print("probability 4 kings shown up =",math.comb(4,4)*math.comb(48,0)/math.comb(52,4))

# Generate X1 and X2 without explicit loops
y = np.random.randint(0, 2, size=(6, num_simulations))

# Calculate X without loops
X = np.sum(y, axis=0)

# Find the frequency of each outcome
unique, counts = np.unique(X, return_counts=True)


# Simulated probability


actual = []
actual.append(prob1)
actual.append(prob2)
actual.append(prob3)
actual.append(prob4)
actual.append(prob5)
X_axis = [0,1,2,3,4]

#theoretical probability
Prob = [0.71873,0.25555,0.02499,0.000709,3.69e-6]

plt.stem(X_axis, actual, markerfmt='o', linefmt='C0-', use_line_collection=True, label='Simulation')
plt.stem(X_axis, Prob, markerfmt='o', linefmt='C1-', use_line_collection=True, label='Theoretical')
plt.xlabel('$k$')  # Use 'k' instead of 'n'
plt.ylabel('$p_{X}(k)$')  # Use 'k' instead of 'n'
plt.legend()
plt.grid()

# Save or display the plot
plt.savefig('/home/aryan/GVV/12.13.3.19/figs/fig.png')
