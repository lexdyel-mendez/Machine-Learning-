import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Reference to what is the sigmoid function
def sigmoid (z):
    return 1/(1+np.exp(-z))

z = np.arange(-5, 5, 0.1)
plt.plot(z, sigmoid(z))
plt.plot([0,0], [0,1], 'r')
plt.plot([-6, 6], [0.5, 0.5], 'r')
plt.show()

#Instance of Data
N = 1000
#Weight (dimensions) of the data
D = 5

X = 5*np.random.randn(N,D)
w = np.random.randn(D, 1)
y = X.dot(w)

y[y<=0] =0
y[y>0] = 1

train_X = X[1:100]
test_X = X[100:]

print(X.shape
