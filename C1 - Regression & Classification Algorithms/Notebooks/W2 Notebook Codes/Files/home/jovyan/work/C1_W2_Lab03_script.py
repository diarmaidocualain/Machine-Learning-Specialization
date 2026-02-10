import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('C1 - Regression & Classification Algorithms/Notebooks/W1 Notebook Codes/Files/home/jovyan/work/deeplearning.mplstyle')
import sys
import numpy.random as rand   
from lab_utils_multi import load_house_data, run_gradient_descent, norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc


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


X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']


# Plot features
#fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
#for i in range(len(ax)):
#    ax[i].scatter(X_train[:,i],y_train)
#    ax[i].set_xlabel(X_features[i])
#ax[0].set_ylabel("Price (1000's)")
#plt.show(block=True)



#set alpha to 9.9e-7
# _, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)


# plot_cost_i_w(X_train, y_train, hist)




mu     = np.mean(X_train,axis=0)   
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma      

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()


# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count"); 
fig.suptitle("distribution of features after normalization")

plt.show()


w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )



#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()



# Unseen test values
# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")


plt_equal_scale(X_train, X_norm, y_train)



print("Done")
