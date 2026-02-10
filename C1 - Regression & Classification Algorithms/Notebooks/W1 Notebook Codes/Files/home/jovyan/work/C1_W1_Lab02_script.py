import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.style.use('C1 - Regression & Classification Algorithms/Notebooks/W1 Notebook Codes/Files/home/jovyan/work/deeplearning.mplstyle')


def compute_model_output(x, w, b):
    """
    Compute the output of the model given input x and parameters w and b.
    
    Args:
        x (ndarray): Input data, shape (m,).
        w (float): Weight parameter.
        b (float): Bias parameter.
    Returns:
        f_wb: Model output, shape (m,).
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    
    return f_wb


x_train  = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
                   
print(f"x_train: {x_train}")
print(f"y_train: {y_train}")


print(f"Number of training examples: {x_train.shape[0]}")
m = x_train.shape[0]

i = 1
x_i = x_train[i]
y_i = y_train[i]
print(f"Training example {i}: (x_i, y_i) = ({x_i}, {y_i})")

# Plot the data
plt.scatter(x_train, y_train, marker='x', c='r', label='Training data')
plt.xlabel('size (1000 sqft)')
plt.ylabel('price (1000s of $)')
plt.title('Housing Prices')
plt.legend()
plt.show()

w = 200.0
b = 100.0
print(f"w: {w}")
print(f"b: {b}")

f_wb = compute_model_output(x_train, w, b)

# Plot the data and the model outptu
plt.scatter(x_train, y_train, marker='x', c='r', label='Training data')
plt.plot(x_train, f_wb, c='b', label='Model output')
plt.xlabel('size (1000 sqft)')
plt.ylabel('price (1000s of $)')
plt.title('Housing Prices and Model Output')
plt.legend()
plt.show()




print("Done!")