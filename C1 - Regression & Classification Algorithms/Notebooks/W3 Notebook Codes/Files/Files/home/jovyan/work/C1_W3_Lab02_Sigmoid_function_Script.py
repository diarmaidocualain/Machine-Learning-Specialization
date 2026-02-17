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
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

z_tmp = np.arange(-10, 11)

y = sigmoid(z_tmp)

print(f"Sigmoid values: {y}")


fig, ax = plt.subplots(1, 1)
ax.plot(z_tmp, y, c="b")
ax.set_title("Sigmoid function")
ax.set_xlabel("y")
ax.set_ylabel("Sigmoid(z)")
ax.grid("on")
plt.show()

x_train = np.array([0., 1, 2, 3, 4, 5, 12.0])
y_train = np.array([0,  0, 0, 1, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(x_train.reshape(-1, 1), y_train)

y_pred = logistic_regression_model.predict(x_train.reshape(-1, 1))

print(f"Predictions: {y_pred}")


fig, ax = plt.subplots(1, 2, figsize=(8,3))

# Plot #1, single variable
pos = (y_train == 1)
neg = (y_train == 0)
ax[0].scatter(x_train[pos], y_train[pos], marker = 'x', s=80, c = 'red', label="y=1")
ax[0].scatter(x_train[neg], y_train[neg], marker = 'o', s=100, c = 'blue', label="y=0")
ax[0].scatter(x_train, y_pred, marker = '.', s=100, c = 'green', label="pred")

ax[0].set_ylim(-0.2,1.2)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()
plt.tight_layout()
plt.show()


print("Done")