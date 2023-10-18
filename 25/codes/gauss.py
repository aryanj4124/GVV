import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load data from the DAT file
filename = "gaussian_data.dat"  # Change to the actual file path if necessary
data = np.loadtxt(filename)

# Sort the data
data.sort()

# Calculate the CDF
cdf = np.arange(1, len(data) + 1) / len(data)

# Fit a normal distribution to the data
mu, std = norm.fit(data)

# Theoretical Gaussian PDF and CDF
x = np.linspace(min(data), max(data), 1000)
pdf_gaussian = norm.pdf(x, mu, std)
cdf_gaussian = norm.cdf(x, mu, std)



# Plot the CDF and theoretical Gaussian CDF
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(data, cdf, color='blue', label="Data CDF")
plt.plot(x, cdf_gaussian, 'r', label="Theoretical Gaussian CDF")
plt.title("Cumulative Distribution Function (CDF)")
plt.xlabel("Data Value")
plt.ylabel("CDF")
plt.legend()

# Plot the PDF and theoretical Gaussian PDF
plt.subplot(1, 2, 2)
plt.hist(data, bins=20, density=True, alpha=0.2, color='g')
plt.plot(x, pdf_gaussian, 'k', linewidth=2, label="Theoretical Gaussian PDF")
plt.title("Probability Density Function (PDF)")
plt.xlabel("Data Value")
plt.ylabel("PDF")
plt.legend()

plt.tight_layout()
plt.savefig("../figs/fig.png")
