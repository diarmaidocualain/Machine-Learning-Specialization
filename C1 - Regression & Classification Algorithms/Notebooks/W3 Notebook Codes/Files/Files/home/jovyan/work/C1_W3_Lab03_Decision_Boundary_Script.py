import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def sigmoid(z):
    """
    Calculate sigmoid function
    
    :param z: Description
    """

    g = 1 / (1 + np.exp(-z))
    return g

# Input is an array. 
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Plot the data
fig, ax = plt.subplots(1, 1, figsize=(8,3))

pos = (y_train == 1)
neg = (y_train == 0)
ax.scatter(X_train[pos][:, 0], X_train[pos][:, 1], marker = 'x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg][:, 0], X_train[neg][:, 1], marker = 'o', s=150, c = 'blue', label="y=0")
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
ax.set_aspect('equal', adjustable='box')
ax.legend()
ax.set_title("Decision boundary")
plt.tight_layout()
plt.show()

# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")
ax.axvline(x=0, color='k', linestyle='--')
ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
ax.grid(True)
plt.show()


# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
ax.scatter(X_train[pos][:, 0], X_train[pos][:, 1], marker = 'x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg][:, 0], X_train[neg][:, 1], marker = 'o', s=150, c = 'blue', label="y=0")
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()




print("Done")