import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('C1 - Regression & Classification Algorithms/Notebooks/W1 Notebook Codes/Files/home/jovyan/work/deeplearning.mplstyle')
import sys
import numpy.random as rand   


def model(x, w, b):
    """
    Model of system (predict)
    Get output f_wb values from input x values using variables of system, w and b. 
    Args:
        x (ndarray (m, )): Input data, m examples
        w (scalar): Weight parameters
        b (scalar): Bias parameter
    Returns:
        f_wb (ndarray (m, )): Output data, m examples 
    """
    f_wb = np.dot(x, w) + b # prediction
    
    return f_wb


def cost(x, y, w, b):
    """
    compute mean square error for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = x.shape[0]
    f_wb = np.dot(x, w) + b 
    total_cost = ((f_wb - y)**2).sum()
    total_cost = total_cost / (2 * m)
    
    return total_cost


def partial_derivatives(x, y, w, b):
    """
    Compute the partial derivates (d / dw) and (d / db)
    Args:
        x (ndarray (m,)): Input data, m examples
        y (ndarray (m,)): Target values, m examples
        w (scalar): Weight parameter
        b (scalar): Bias parameter
    Returns:
        d / dw and d/db
    """
    f_wb = model(x, w, b)
    m = x.shape[0]
    d_dw = (1 / m) * (np.dot(x.T, (f_wb - y)))
    d_db = (1 / m) * ((f_wb - y).sum())

    return (d_dw, d_db)


def gradient_descent(x, y, w, b, a = 0.1, min_delta = 0.01, max_iter = 1000):
    """
    Performs gradient descent to optimize values of w, b
    Args:
        x (ndarray (m,)): Input data, m examples
        y (ndarray (m,)): Target values, m examples
        w (scalar): Weight parameter
        b (scalar): Bias parameter
        a (scalar): alpha, the rate that we update at. 
        delta (scalar): when the difference from current to previous value is less than delta, exit
        max_iter (scalar): when the number of iterations reaches this value, exit
    Returns:
        new_w, new_b 
    """
    delta = np.finfo(np.float64).max # Max  float num in python
    iter = 0
    prev_total_cost = 0.0
    cost_history = np.zeros(max_iter)
    while (delta > min_delta and iter < max_iter): 
        (d_dw, d_db) = partial_derivatives(x, y, w, b)
        w = w - (a * d_dw)
        b = b - (a * d_db)
        total_cost = cost(x, y, w, b)
        cost_history[iter] = total_cost
        delta = np.abs(total_cost - prev_total_cost)
        w_str = ", ".join([f"{val:.2f}" for val in w])
        print(f"{iter}: New w values are: {w_str} and b value: {b:.2f} give a total cost of {total_cost:.4f}")
        prev_total_cost = total_cost
        iter += 1
    
    return (w, b, cost_history)


def z_score_normalisation(data):
    """
    Performs z score normalisation on a training dataset
    
    data: 
    """
    data_mu = data.mean(axis=0)
    data_sigma = data.std(axis=0)
    data_std = (data - data_mu) / data_sigma
    return (data_std, data_mu, data_sigma)


def z_score_unnormalisation(data_std, data_mu, data_sigma):
    """
    Performs z score normalisation on a training dataset
    
    data: 
    """
    data = (data_std * data_sigma) + data_mu
    return data


# | Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
# | ----------------| ------------------- |----------------- |--------------|-------------- |  
# | 2104            | 5                   | 1                | 45           | 460           |  
# | 1416            | 3                   | 2                | 40           | 232           |  
# | 852             | 2                   | 1                | 35           | 178           |  


x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])

y_train = np.array([460,232,178])

(x_std, x_mu, x_sigma) = z_score_normalisation(x_train)
(y_std, y_mu, y_sigma) = z_score_normalisation(y_train)

# Initialise the weights and the bias
# w = np.zeros([1, x_train.shape[1]])
# b = 0.0
# b = 785.1811367994083
w = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

cost_history = 0

# initialize parameters
initial_w = np.zeros_like(w)
initial_b = 0.
# some gradient descent settings
iterations = 10000
alpha = 0.1



# Compute and display cost using our pre-chosen optimal parameters. 
# overall_cost = cost(x_train, y_train, w, b)
# print(f'Cost at optimal w : {overall_cost}')



(w, b, cost_history) = gradient_descent(x_std, y_std, initial_w, initial_b, a = alpha, min_delta = 0.0, max_iter = iterations)


# Check predictions using the w and b you just found
for i in range(x_train.shape[0]):
    prediction_std = np.dot(x_std[i], w) + b
    prediction = z_score_unnormalisation(prediction_std, y_mu, y_sigma)
    print(f"Prediction: {prediction:.2f}, Target: {y_train[i]}")

print("Done!")