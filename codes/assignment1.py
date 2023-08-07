import numpy as np

A = np.array([[1],[-1]])
B = np.array([[-4],[6]])
m = B - A
n = np.array([[(m[1][0]),-m[0][0]]])
c = np.dot(n,A)
eqn = f"({n[0][0]} {n[0][1]})x = {c[0][0]}"
print("The normal form of the equation of AB is:",eqn)



