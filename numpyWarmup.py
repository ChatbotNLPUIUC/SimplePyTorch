import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10
#N:batch size, D_in:input dim, H:hidden dim, D_out:output dim

#Random input and random output
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

#initializing weights random
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    #Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)