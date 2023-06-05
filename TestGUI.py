import matplotlib.pyplot as plt
import numpy as np

x = range(0, 100)
y = [t**2 for t in x]

pic = plt.figure(figsize=(12, 8))
plt.title("Test", fontsize=15)
plt.plot(x, y, "r-")
plt.show()