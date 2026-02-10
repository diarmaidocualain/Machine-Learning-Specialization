import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('C1 - Regression & Classification Algorithms/Notebooks/W1 Notebook Codes/Files/home/jovyan/work/deeplearning.mplstyle')
import sys
import numpy.random as rand   


# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4)
print(f"np.zeros(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.zeros((4,))
print(f"np.zeros(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.random_sample(4)
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# Numpy routines which allocate memory and fill arrays with value but do not accept share as input argument
a = np.arange(4.)
print(f"np.arange(4.): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.rand(4)
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2])
print(f"np.array([5,4,3,2]):  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.array([5.,4,3,2])
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# vector indexing operations on 1D vectors
a =- np.arange(10)

# access an element
print(f"a[2].shape: {a[2].shape} a[2] = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end 
print(f"a[-1] {a[-1]}")

# indexes must be within the range of the vector or they will produce an error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# vector slicing operations
a = np.arange(10)
print(f"a = {a}")

# access 5 consecutive elements (start:stop:step)
c = a[2:7:1]; print(f"c = a[2:7:1] {c}")

# access 3 elements separated by two 
c = a[2:7:2]; print(f"c = a[2:7:2] {c}")

# access all elements index 3 and above
c = a[3::1]; print(f"c = a[3::1] {c}")

# access all elements below index 3
c = a[:3:1]; print(f"c = a[:3:1] {c}")

# access all elements
c = a[::]; print(f"c = a[::] {c}")


# Single vector operations
a = np.array([1, 2, 3, 4])
print(f"a {a}")

# negate elements of a
print(f"-a {-a}")

# sum all elements of a and return a scalar
print(f"a.sum() {a.sum()}")

# mean all elements of a and return a scalar
print(f"a.mean() {a.mean()}")

# Sqauare all elements of a and return a scalar
print(f"a**2 {a**2}")

# Vector vector element-wise operations
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

# try a mismatched vector operation
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# Scalar vector operations
a = np.array([1, 2, 3, 4])

# multiply a by a scalar
b = 5 * a 
print(f"b = 5 * a : {b}")

# Vector Vector dot product

def my_dot(a, b):
    """
    Compute the dot product of two vector
    
    Args:
    :param a (ndarray): 
    :param b (ndarray): 
    """

    x = 0
    a_shape = a.shape[0]
    for i in range(a_shape):
        x = x + a[i] * b[i]

    return(x)

# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")

c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

# 3.4.7 Need for speed
import time
np.random.seed(1)

a = np.random.rand(1000000) # very large arrays
b = np.random.rand(1000000)

tic = time.time() # capture start time in ns
c = np.dot(a, b)
toc = time.time() # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory

# 4.3 Matrix creation
a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}") 

# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")


#vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na = {a}")

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

# Access a raow
print(f"a[2].shape {a[2].shape}")



#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")









print()





print("Done!")