import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


x_train1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0])
y_train1 = np.array([0, 0, 0, 1, 1, 1, 1])

X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])


pos1 = (y_train1 == 1)
neg1 = (y_train1 == 0)

pos2 = (y_train2 == 1)
neg2 = (y_train2 == 0)


fig, ax = plt.subplots(1, 2, figsize=(8,3))

# Plot #1, single variable
ax[0].scatter(x_train1[pos1], y_train1[pos1], marker = 'x', s=80, c = 'red', label="y=1")
ax[0].scatter(x_train1[neg1], y_train1[neg1], marker = 'o', s=100, c = 'blue', label="y=0")

ax[0].set_ylim(-0.08,1.1)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()

#plot 2, two variables
ax[1].scatter(X_train2[pos2][:, 0], X_train2[pos2][:, 1], marker = 'x', s=80, c = 'red', label="y=1")
ax[1].scatter(X_train2[neg2][:, 0], X_train2[neg2][:, 1], marker = 'o', s=100, c = 'blue', label="y=0")
# plot_data(X_train2, y_train2, ax[1])
ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$x_1$', fontsize=12)
ax[1].set_xlabel('$x_0$', fontsize=12)
ax[1].set_title('two variable plot')
ax[1].legend()
plt.tight_layout()
plt.show()

linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train1.reshape(-1, 1), y_train1)

y_pred = linear_regression_model.predict(x_train1.reshape(-1, 1))

print(f"Predictions: {y_pred}")

# Add a threshold
y_pred = (y_pred > 0.5)


fig, ax = plt.subplots(1, 2, figsize=(8,3))

# Plot #1, single variable
ax[0].scatter(x_train1[pos1], y_train1[pos1], marker = 'x', s=80, c = 'red', label="y=1")
ax[0].scatter(x_train1[neg1], y_train1[neg1], marker = 'o', s=100, c = 'blue', label="y=0")
ax[0].scatter(x_train1, y_pred, marker = '.', s=100, c = 'green', label="pred")

ax[0].set_ylim(-0.2,1.2)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()
plt.tight_layout()
plt.show()

print("Done")