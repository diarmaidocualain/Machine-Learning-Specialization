import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def model(X_data, w, b):
    f_wb = np.dot(X_data, w) + b
    return f_wb


def cost(X_data, w, b, y_data):
    m = X_data.shape[0]
    f_wb = model(X_data, w, b)
    total_cost = np.sum((f_wb - y_data) ** 2)
    total_cost = total_cost / (2 * m)
    return total_cost


def partial_derivatives(X_data, w, b, y_data):
    m = X_data.shape[0]
    f_wb = model(X_data, w, b)
    d_dw = (1 / m) * np.dot(X_data.T, (f_wb - y_data))
    d_db = (1 / m) * (f_wb - y_data).sum()

    return (d_dw, d_db)


def gradient_descent(X_data, w, b, y_data, alpha = 0.1, max_iter = 1000000, max_delta = 0.00001):

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


def z_score_unnormalisation(data_std, data_mu, data_sigma):
    """
    Performs z score normalisation on a training dataset
    
    data: 
    """
    data = (data_std * data_sigma) + data_mu
    return data


# # Read in the csv dataset. 
# data = np.loadtxt('/home/diarmaid/Documents/learning/coursera/machine_learning_specialization/Machine-Learning-Specialization/C1 - Regression & Classification Algorithms/Notebooks/W2 Notebook Codes/Files/home/jovyan/work/data/houses.txt', delimiter=",")
# (data_norm, mu, sigma) = zscore_normalize_features(data)
# X_data = data_norm[:, 0:4]
# y_data = data_norm[:, -1]
# 
# x_feature_names = {"size(sqft)", "bedrooms", "floors", "age"}
# y_feature_names = {"cost"}
# 
# 
# # Plot each feature against the price of the house and visually see if there is any correlation
# 
# plt.plot(X_data[:, 0], y_data, 'x')
# plt.xlabel('size(sqft)')
# plt.ylabel('cost')
# plt.title("Size vs cost")
# plt.show(block=True)
# 
# w = np.zeros(X_data.shape[1])
# b = 0.0
# 
# (w, b, cost_history) = gradient_descent(X_data, w, b, y_data, alpha = 0.04, max_iter = 1000, max_delta = 0.0001)
# 
# plt.plot(range(cost_history.shape[0]), cost_history)
# plt.ylabel('cost')
# plt.xlabel('iteration')
# plt.title("cost_history")
# plt.show(block=True)
# 
# # Using the training data, plot the predictions over the target values
# f_wb = model(X_data, w, b)
# plt.plot(X_data[:, 0], y_data, 'o')
# plt.plot(X_data[:, 0], f_wb, 'x')
# plt.xlabel('size(sqft)')
# plt.ylabel('cost')
# plt.title("Size vs cost")
# plt.show(block=True)

# Create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x ** 2
X = X.reshape(-1, 1)

# plt.plot(x, y, 'x')
# plt.xlabel("X")
# plt.ylabel("y")
# plt.title("X vs y")
# plt.show(block=True)

(data_norm, mu, sigma) = zscore_normalize_features(np.column_stack([X, y]))
X = data_norm[:, 0]
y = data_norm[:, 1]

w = 0.0
b = 0.0

(w, b, cost_history) = gradient_descent(X, w, b, y, alpha = 0.04, max_iter = 1000, max_delta = 0.0001)

# Using the training data, plot the predictions over the target values
f_wb = model(X, w, b)



# Un-normalise the data
data = z_score_unnormalisation(np.column_stack([X, f_wb]), mu, sigma)
X = data[:, 0]
f_wb = data[:, 1]
data = z_score_unnormalisation(np.column_stack([X, y]), mu, sigma)
X = data[:, 0]
y = data[:, 1]

plt.plot(x, y, 'x')
plt.plot(x, f_wb, 'o')
plt.xlabel("X")
plt.ylabel("y")
plt.title("X vs y")
plt.show(block=True)



print("Done")