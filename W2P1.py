def Ex2():
    import math
    print("Ex")
    # GRADED FUNCTION: basic_sigmoid

    def basic_sigmoid(x):
        """
        Compute sigmoid of x.

        Arguments:
        x -- A scalar

        Return:
        s -- sigmoid(x)
        """
        # (≈ 1 line of code)
        # s =
        # YOUR CODE STARTS HERE
        s = 1 / (1 + math.exp(-x))

        # YOUR CODE ENDS HERE

        return s

    # In[6]:

    print("basic_sigmoid(1) = " + str(basic_sigmoid(1)))


    # x = [1, 2, 3]
    # basic_sigmoid(x)  # Error

    import numpy as np

    # example of np.exp
    t_x = np.array([1, 2, 3])
    print(np.exp(t_x))  # result is (exp(1), exp(2), exp(3))

    # example of vector operation
    t_x = np.array([1, 2, 3])
    print(t_x + 3)

def Ex3():





# <a name='1-2'></a>
# ### 1.2 - Sigmoid Gradient
# 
# As you've seen in lecture, you will need to compute gradients to optimize loss functions using backpropagation. Let's code your first gradient function.
# 
# <a name='ex-4'></a>
# ### Exercise 4 - sigmoid_derivative
# Implement the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x. The formula is: 
# 
# $$sigmoid\_derivative(x) = \sigma'(x) = \sigma(x) (1 - \sigma(x))\tag{2}$$
# 
# You often code this function in two steps:
# 1. Set s to be the sigmoid of x. You might find your sigmoid(x) function useful.
# 2. Compute $\sigma'(x) = s(1-s)$

# In[12]:


# GRADED FUNCTION: sigmoid_derivative

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """

    # (≈ 2 lines of code)
    # s = 
    # ds = 
    # YOUR CODE STARTS HERE
    s = sigmoid(x)
    ds = s * (1 - s)
    # YOUR CODE ENDS HERE

    return ds


# In[13]:


t_x = np.array([1, 2, 3])
print("sigmoid_derivative(t_x) = " + str(sigmoid_derivative(t_x)))

sigmoid_derivative_test(sigmoid_derivative)


# <a name='1-3'></a>
# ### 1.3 - Reshaping arrays ###
# 
# Two common numpy functions used in deep learning are [np.shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html) and [np.reshape()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html). 
# - X.shape is used to get the shape (dimension) of a matrix/vector X. 
# - X.reshape(...) is used to reshape X into some other dimension. 
# 
# For example, in computer science, an image is represented by a 3D array of shape $(length, height, depth = 3)$. However, when you read an image as the input of an algorithm you convert it to a vector of shape $(length*height*3, 1)$. In other words, you "unroll", or reshape, the 3D array into a 1D vector.
# 
# <img src="images/image2vector_kiank.png" style="width:500px;height:300;">
# 
# <a name='ex-5'></a>
# ### Exercise 5 - image2vector
# Implement `image2vector()` that takes an input of shape (length, height, 3) and returns a vector of shape (length\*height\*3, 1). For example, if you would like to reshape an array v of shape (a, b, c) into a vector of shape (a*b,c) you would do:
# ``` python
# v = v.reshape((v.shape[0] * v.shape[1], v.shape[2])) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
# ```
# - Please don't hardcode the dimensions of image as a constant. Instead look up the quantities you need with `image.shape[0]`, etc. 
# - You can use v = v.reshape(-1, 1). Just make sure you understand why it works.

# In[14]:


# GRADED FUNCTION:image2vector

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    # (≈ 1 line of code)
    # v =
    # YOUR CODE STARTS HERE
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))

    # YOUR CODE ENDS HERE

    return v


# In[15]:


# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
t_image = np.array([[[0.67826139, 0.29380381],
                     [0.90714982, 0.52835647],
                     [0.4215251, 0.45017551]],

                    [[0.92814219, 0.96677647],
                     [0.85304703, 0.52351845],
                     [0.19981397, 0.27417313]],

                    [[0.60659855, 0.00533165],
                     [0.10820313, 0.49978937],
                     [0.34144279, 0.94630077]]])

print("image2vector(image) = " + str(image2vector(t_image)))

image2vector_test(image2vector)


# <a name='1-4'></a>
# ### 1.4 - Normalizing rows
# 
# Another common technique we use in Machine Learning and Deep Learning is to normalize our data. It often leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to $ \frac{x}{\| x\|} $ (dividing each row vector of x by its norm).
# 
# For example, if 
# $$x = \begin{bmatrix}
#         0 & 3 & 4 \\
#         2 & 6 & 4 \\
# \end{bmatrix}\tag{3}$$ 
# then 
# $$\| x\| = \text{np.linalg.norm(x, axis=1, keepdims=True)} = \begin{bmatrix}
#     5 \\
#     \sqrt{56} \\
# \end{bmatrix}\tag{4} $$
# and
# $$ x\_normalized = \frac{x}{\| x\|} = \begin{bmatrix}
#     0 & \frac{3}{5} & \frac{4}{5} \\
#     \frac{2}{\sqrt{56}} & \frac{6}{\sqrt{56}} & \frac{4}{\sqrt{56}} \\
# \end{bmatrix}\tag{5}$$ 
# 
# Note that you can divide matrices of different sizes and it works fine: this is called broadcasting and you're going to learn about it in part 5.
# 
# With `keepdims=True` the result will broadcast correctly against the original x.
# 
# `axis=1` means you are going to get the norm in a row-wise manner. If you need the norm in a column-wise way, you would need to set `axis=0`. 
# 
# numpy.linalg.norm has another parameter `ord` where we specify the type of normalization to be done (in the exercise below you'll do 2-norm). To get familiar with the types of normalization you can visit [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
# 
# <a name='ex-6'></a>
# ### Exercise 6 - normalize_rows
# Implement normalizeRows() to normalize the rows of a matrix. After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).
# 
# **Note**: Don't try to use `x /= x_norm`. For the matrix division numpy must broadcast the x_norm, which is not supported by the operant `/=`

# In[16]:


# GRADED FUNCTION: normalize_rows

def normalize_rows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """

    # (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    # x_norm =
    # Divide x by its norm.
    # x =
    # YOUR CODE STARTS HERE
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm

    # YOUR CODE ENDS HERE

    return x


# In[17]:


x = np.array([[0, 3, 4],
              [1, 6, 4]])
print("normalizeRows(x) = " + str(normalize_rows(x)))

normalizeRows_test(normalize_rows)


# **Note**:
# In normalize_rows(), you can try to print the shapes of x_norm and x, and then rerun the assessment. You'll find out that they have different shapes. This is normal given that x_norm takes the norm of each row of x. So x_norm has the same number of rows but only 1 column. So how did it work when you divided x by x_norm? This is called broadcasting and we'll talk about it now! 

# <a name='ex-7'></a>
# ### Exercise 7 - softmax
# Implement a softmax function using numpy. You can think of softmax as a normalizing function used when your algorithm needs to classify two or more classes. You will learn more about softmax in the second course of this specialization.
# 
# **Instructions**:
# - $\text{for } x \in \mathbb{R}^{1\times n} \text{,     }$
# 
# \begin{align*}
#  softmax(x) &= softmax\left(\begin{bmatrix}
#     x_1  &&
#     x_2 &&
#     ...  &&
#     x_n  
# \end{bmatrix}\right) \\&= \begin{bmatrix}
#     \frac{e^{x_1}}{\sum_{j}e^{x_j}}  &&
#     \frac{e^{x_2}}{\sum_{j}e^{x_j}}  &&
#     ...  &&
#     \frac{e^{x_n}}{\sum_{j}e^{x_j}} 
# \end{bmatrix} 
# \end{align*}
# 
# - $\text{for a matrix } x \in \mathbb{R}^{m \times n} \text{,  $x_{ij}$ maps to the element in the $i^{th}$ row and $j^{th}$ column of $x$, thus we have: }$  
# 
# \begin{align*}
# softmax(x) &= softmax\begin{bmatrix}
#             x_{11} & x_{12} & x_{13} & \dots  & x_{1n} \\
#             x_{21} & x_{22} & x_{23} & \dots  & x_{2n} \\
#             \vdots & \vdots & \vdots & \ddots & \vdots \\
#             x_{m1} & x_{m2} & x_{m3} & \dots  & x_{mn}
#             \end{bmatrix} \\ \\&= 
#  \begin{bmatrix}
#     \frac{e^{x_{11}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{12}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{13}}}{\sum_{j}e^{x_{1j}}} & \dots  & \frac{e^{x_{1n}}}{\sum_{j}e^{x_{1j}}} \\
#     \frac{e^{x_{21}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{22}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{23}}}{\sum_{j}e^{x_{2j}}} & \dots  & \frac{e^{x_{2n}}}{\sum_{j}e^{x_{2j}}} \\
#     \vdots & \vdots & \vdots & \ddots & \vdots \\
#     \frac{e^{x_{m1}}}{\sum_{j}e^{x_{mj}}} & \frac{e^{x_{m2}}}{\sum_{j}e^{x_{mj}}} & \frac{e^{x_{m3}}}{\sum_{j}e^{x_{mj}}} & \dots  & \frac{e^{x_{mn}}}{\sum_{j}e^{x_{mj}}}
# \end{bmatrix} \\ \\ &= \begin{pmatrix}
#     softmax\text{(first row of x)}  \\
#     softmax\text{(second row of x)} \\
#     \vdots  \\
#     softmax\text{(last row of x)} \\
# \end{pmatrix} 
# \end{align*}

# **Notes:**
# Note that later in the course, you'll see "m" used to represent the "number of training examples", and each training example is in its own column of the matrix. Also, each feature will be in its own row (each row has data for the same feature).  
# Softmax should be performed for all features of each training example, so softmax would be performed on the columns (once we switch to that representation later in this course).
# 
# However, in this coding practice, we're just focusing on getting familiar with Python, so we're using the common math notation $m \times n$  
# where $m$ is the number of rows and $n$ is the number of columns.

# In[18]:


# GRADED FUNCTION: softmax
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (m,n).

    Argument:
    x -- A numpy matrix of shape (m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """

    # (≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    # x_exp = ...

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    # x_sum = ...

    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    # s = ...

    # YOUR CODE STARTS HERE
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    # YOUR CODE ENDS HERE

    return s


# In[19]:


t_x = np.array([[9, 2, 5, 0, 0],
                [7, 5, 0, 0, 0]])
print("softmax(x) = " + str(softmax(t_x)))

softmax_test(softmax)

# #### Notes
# - If you print the shapes of x_exp, x_sum and s above and rerun the assessment cell, you will see that x_sum is of shape (2,1) while x_exp and s are of shape (2,5). **x_exp/x_sum** works due to python broadcasting.
# 
# Congratulations! You now have a pretty good understanding of python numpy and have implemented a few useful functions that you will be using in deep learning.

# <font color='blue'>
# <b>What you need to remember:</b>
#     
# - np.exp(x) works for any np.array x and applies the exponential function to every coordinate
# - the sigmoid function and its gradient
# - image2vector is commonly used in deep learning
# - np.reshape is widely used. In the future, you'll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs. 
# - numpy has efficient built-in functions
# - broadcasting is extremely useful

# <a name='2'></a>
# ## 2 - Vectorization
# 
# 
# In deep learning, you deal with very large datasets. Hence, a non-computationally-optimal function can become a huge bottleneck in your algorithm and can result in a model that takes ages to run. To make sure that your code is  computationally efficient, you will use vectorization. For example, try to tell the difference between the following implementations of the dot/outer/elementwise product.

# In[45]:


import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1), len(x2)))  # we create a len(x1)*len(x2) matrix with only zeros

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i] * x2[j]
toc = time.process_time()
print("outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))

for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
toc = time.process_time()
print("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3, len(x1))  # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])

for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i, j] * x1[j]
toc = time.process_time()
print("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

# In[46]:


x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()
print("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1, x2)
toc = time.process_time()
print("outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1, x2)
toc = time.process_time()
print("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
dot = np.dot(W, x1)
toc = time.process_time()
print("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")


# As you may have noticed, the vectorized implementation is much cleaner and more efficient. For bigger vectors/matrices, the differences in running time become even bigger. 
# 
# **Note** that `np.dot()` performs a matrix-matrix or matrix-vector multiplication. This is different from `np.multiply()` and the `*` operator (which is equivalent to  `.*` in Matlab/Octave), which performs an element-wise multiplication.

# <a name='2-1'></a>
# ### 2.1 Implement the L1 and L2 loss functions
# 
# <a name='ex-8'></a>
# ### Exercise 8 - L1 
# Implement the numpy vectorized version of the L1 loss. You may find the function abs(x) (absolute value of x) useful.
# 
# **Reminder**:
# - The loss is used to evaluate the performance of your model. The bigger your loss is, the more different your predictions ($ \hat{y} $) are from the true values ($y$). In deep learning, you use optimization algorithms like Gradient Descent to train your model and to minimize the cost.
# - L1 loss is defined as:
# $$\begin{align*} & L_1(\hat{y}, y) = \sum_{i=0}^{m-1}|y^{(i)} - \hat{y}^{(i)}| \end{align*}\tag{6}$$

# In[47]:


# GRADED FUNCTION: L1

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """

    # (≈ 1 line of code)
    # loss = 
    # YOUR CODE STARTS HERE
    loss = np.sum(abs(y - yhat))

    # YOUR CODE ENDS HERE

    return loss


# In[48]:


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat, y)))

L1_test(L1)


# <a name='ex-9'></a>
# ### Exercise 9 - L2
# Implement the numpy vectorized version of the L2 loss. There are several way of implementing the L2 loss but you may find the function np.dot() useful. As a reminder, if $x = [x_1, x_2, ..., x_n]$, then `np.dot(x,x)` = $\sum_{j=1}^n x_j^{2}$. 
# 
# - L2 loss is defined as $$\begin{align*} & L_2(\hat{y},y) = \sum_{i=0}^{m-1}(y^{(i)} - \hat{y}^{(i)})^2 \end{align*}\tag{7}$$

# In[55]:


# GRADED FUNCTION: L2

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """

    # (≈ 1 line of code)
    # loss = ...
    # YOUR CODE STARTS HERE
    loss = np.dot(y - yhat, y - yhat)

    # YOUR CODE ENDS HERE

    return loss


# In[56]:


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

print("L2 = " + str(L2(yhat, y)))

L2_test(L2)

# Congratulations on completing this assignment. We hope that this little warm-up exercise helps you in the future assignments, which will be more exciting and interesting!

# <font color='blue'>
# <b>What to remember:</b>
#     
# - Vectorization is very important in deep learning. It provides computational efficiency and clarity.
# - You have reviewed the L1 and L2 loss.
# - You are familiar with many numpy functions such as np.sum, np.dot, np.multiply, np.maximum, etc...
