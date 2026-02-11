import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



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



X_train = np.array([1.0, 2.0]) # features
y_train = np.array([300, 500]) # target

linear_model = LinearRegression()

# X must be a 2-D matrix
linear_model.fit(X_train.reshape(-1, 1), y_train)



b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
test_value = 1200
print(f"'manual' prediction: f_wb = wx+b : {test_value*w + b}")

y_pred = linear_model.predict(X_train.reshape(-1, 1))

print("Prediction on training set:", y_pred)

X_test = np.array([[test_value]])
print(f"Prediction for {test_value} sqft house: ${linear_model.predict(X_test)[0]:0.2f}")





# Read in the csv dataset. 
data = np.loadtxt('/home/diarmaid/Documents/learning/coursera/machine_learning_specialization/Machine-Learning-Specialization/C1 - Regression & Classification Algorithms/Notebooks/W2 Notebook Codes/Files/home/jovyan/work/data/houses.txt', delimiter=",")
# (data_norm, mu, sigma) = zscore_normalize_features(data)
# X_data = data_norm[:, 0:4]
# y_data = data_norm[:, -1]
X_train = data[:, 0:-1]
y_train = data[:, -1]
# 
x_feature_names = {"size(sqft)", "bedrooms", "floors", "age"}
y_feature_names = {"cost"}
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


linear_model = LinearRegression()
linear_model.fit(X_train, y_train) 

b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")




print(f"Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
print(f"prediction using w,b:\n {(X_train @ w + b)[:4]}")
print(f"Target values \n {y_train[:4]}")

x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
x_house_predict = linear_model.predict(x_house)[0]
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.2f}")



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



print("Done")