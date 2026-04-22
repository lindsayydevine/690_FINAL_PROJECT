import numpy as np
data = np.load('features/my_features.npz')
X = data['features']   # shape (N, 1028)
y = data['labels']     # shape (N,)
pids = data['pids']    # shape (N,)

print(X.shape) # should be (689070, 1028)
print(y.shape) # should be (689070,)
print(pids.shape) # should be (689070,)