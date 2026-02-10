import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('C1 - Regression & Classification Algorithms/Notebooks/W1 Notebook Codes/Files/home/jovyan/work/deeplearning.mplstyle')
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl


def model(x, w, b):
    """
    Model of system
    Get output f_wb values from input x values using variables of system, w and b. 
    Args:
        x (ndarray (m, )): Input data, m examples
        w (scalar): Weight parameters
        b (scalar): Bias parameter
    Returns:
        f_wb (ndarray (m, )): Output data, m examples 
    """
    f_wb = w * x + b # prediction
    
    return f_wb


def cost(x, y, w, b):
    """
    Compute the mean square cost for linear regression
    Args:
        x (ndarray (m,)): Input data, m examples
        y (ndarray (m,)): Target values, m examples
        w (scalar): Weight parameter
        b (scalar): Bias parameter
    Returns:
        total_cost (scalar): The total mean square cost
    """
    f_wb = model(x, w, b)
    costs = (f_wb - y) ** 2 # Square the error for each point
    m = x.shape[0] # number of samples
    total_cost = (1 / (2 * m)) * costs.sum()

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
    d_dw = (1 / m) * (((f_wb - y) * x).sum())
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
    delta = np.finfo(np.float64).max
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
        print(f"{iter}:New w and b values are: {w:.2f}, {b:.2f} and give a total cost of {total_cost:.4f}")
        prev_total_cost = total_cost
        iter += 1
    
    return (w, b, cost_history)
        






# def compute_cost(x, y, w, b):
#     """
#     Compute the cost for linear regression
#     Args:
#         x (ndarray (m,)): Input data, m examples
#         y (ndarray (m,)): Target values, m examples
#         w (scalar): Weight parameter
#         b (scalar): Bias parameter
#     Returns:
#         total_cost (scalar): The total cost
#         """
#     m = x.shape[0] # number of training examples
#     cost_sum = 0.0
#     for i in range(m):
#         f_wb = w * x[i] + b # prediction
#         cost = (f_wb - y[i]) ** 2 # squared error
#         cost_sum += cost
#     
#     total_cost = (1 / (2 * m)) * cost_sum
#     return total_cost


x_train = np.array([1.0, 2.0]) # size in 1000 sq ft
y_train = np.array([300.0, 500.0]) # price in 1000s of dollars


w = 200
b = 100

total_cost = cost(x_train, y_train, w, b)
print(f"Cost is: {total_cost}")

#plt_intuition(x_train, y_train)

(d_dw, d_db) = partial_derivatives(x_train, y_train, w, b)

print(f"Partial derivatives are {d_dw} d_dw and {d_db} d_db")

w = 210
b = 180

(w, b, cost_history) = gradient_descent(x_train, y_train, w, b)



print(f" New w, b values are: {w, b}")

plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost over time")
plt.grid(True)
plt.show()

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730 ])

fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)


soup_bowl()






print("Done!")