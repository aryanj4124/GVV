import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-5,5)
y = (-7/5)*(x) + (2/5)

plt.title("Line AB")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.plot(x,y)
plt.savefig("figure1.png")
