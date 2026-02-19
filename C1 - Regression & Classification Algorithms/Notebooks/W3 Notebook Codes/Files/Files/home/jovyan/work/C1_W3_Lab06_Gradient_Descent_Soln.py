import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from lab_utils_common import  plt_tumor_data
from plt_quad_logistic import plt_quad_logistic, plt_prob


def sigmoid(z):
    """
    Calculate sigmoid function
    
    :param z: Description
    """

    g = 1 / (1 + np.exp(-z))
    return g


def model(X_data, w, b):
    # Logistic Regression
    f_wb = sigmoid(np.dot(X_data, w) + b)

    # Linear Regression
    #f_wb = np.dot(X_data, w) + b
    return f_wb


def cost(X_data, w, b, y):
    m = X_data.shape[0]
    f_wb = model(X_data, w, b)

    # Linear regression
    # total_cost = np.sum((f_wb - y_data) ** 2)
    # total_cost = total_cost / (2 * m)

    # Logistic Regression
    cost_sum = 0.0
    
    for i in range(0, m):
        f_wb = model(X_data[i], w, b)
        cost_sum += -(y[i]) * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
    
    total_cost = cost_sum / m
    return total_cost


def partial_derivatives(X_data, w, b, y_data):
    m = X_data.shape[0]
    f_wb = model(X_data, w, b)
    d_dw = (1 / m) * np.dot(X_data.T, (f_wb - y_data))
    d_db = (1 / m) * (f_wb - y_data).sum()

    return (d_dw, d_db)


def gradient_descent(X_data, w, b, y_data, alpha = 0.1, max_iter = 10000, max_delta = 0.000001):

    delta = np.finfo(np.float64).max # Max  float num in python
    iter_num = 0
    cost_history = np.zeros(max_iter)
    prev_total_cost = 0.0

    while iter_num < max_iter and delta > max_delta:
        (d_dw, d_db) = partial_derivatives(X_data, w, b, y_data)
        w = w - (alpha * d_dw)
        b = b - (alpha * d_db)
        total_cost = cost(X_data, w, b, y_data)
        cost_history[iter_num] = total_cost
        delta = abs(total_cost - prev_total_cost)
        prev_total_cost = total_cost
        w_string = np.array2string(w, precision=2, suppress_small=True)
        b_string = np.array2string(b, precision=2, suppress_small=True)
        print(f"Iter {iter_num}: w: {w_string}, b: {b_string}, cost: {total_cost}")
        iter_num += 1

    return(w, b, cost_history[0:iter_num])


def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)







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

# Compute the cost using example values
w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.

w_out, b_out, cost_history = gradient_descent(X_train, w_tmp, b_tmp, y_train);
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

# Plot the data
fig,ax = plt.subplots(1,1,figsize=(5,4))
pos = (y_train == 1)
neg = (y_train == 0)
ax.scatter(X_train[pos][:, 0], X_train[pos][:, 1], marker = 'x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg][:, 0], X_train[neg][:, 1], marker = 'o', s=150, c = 'blue', label="y=0")
ax.axis([0, 4, 0, 3.5])
# Plot the decision boundary
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c="blue", lw=1)
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
ax.set_aspect('equal', adjustable='box')
ax.legend()
ax.set_title("Decision boundary")
plt.tight_layout()
plt.show()


# Single variable example. 

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])



fig,ax = plt.subplots(1,1,figsize=(4,3))
plt_tumor_data(x_train, y_train, ax)
plt.show()




w_range = np.array([-1, 7])
b_range = np.array([1, -14])
quad = plt_quad_logistic( x_train, y_train, w_range, b_range )











print("Done")