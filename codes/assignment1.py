import numpy as np

A = np.array([[1],[-1]])
B = np.array([[-4],[6]])
X = np.array([["x"],["y"]])
m = B - A
n = np.array([[(m[1][0]),-m[0][0]]])
c = np.dot(n,A)
eqn = f"{n[0][0]}x + {n[0][1]}y = {c[0][0]}"
print(eqn)



