import numpy as np

# 1) Creating arrays
arr = np.array([1, 2, 3])            # from Python list
a = np.array([1, 2, 3])            # from Python list
b = np.arange(0, 10, 2)           # [0,2,4,6,8]
c = np.linspace(0, 1, 5)          # 5 values evenly spaced between 0 and 1
z = np.zeros((3,4))               # shape (3,4) filled with 0.0
o = np.ones((2,2), dtype=int)     # integer ones
I = np.eye(4)                     # 4x4 identity matrix
r = np.random.default_rng().random((3,3))  # random floats in [0,1)

# 2) dtypes & conversions
a.dtype                # e.g. dtype('int64')
a.astype(float)        # convert type (returns new array)
np.array([1.0, 2.0], dtype=np.int32)  # create w/ dtype

# 3) Shape, ndim, size
x = np.arange(12).reshape(3,4)
x.shape   # (3,4)
x.ndim    # 2
x.size    # 12

# 4) Indexing & slicing (views, not copies usually)
row0 = x[0]        # first row
elem = x[1,2]      # single element
slice = x[:, 1:3]  # all rows, cols 1 and 2
slice[0,0] = 999   # modifies original x (view)

# 5) Fancy indexing & boolean masks
y = np.array([10,20,30,40,50])
idx = [0,2,4]
y[idx]             # -> [10,30,50]  (creates copy)

mask = y > 25
y[mask]            # -> [30,40,50]
y[y % 20 == 0]     # chaining conditions

# 6) Broadcasting (powerful)
A = np.array([[1,2,3],[4,5,6]])   # shape (2,3)
v = np.array([10,20,30])          # shape (3,)
A + v                              # v broadcast across rows
A + 1                              # scalar broadcast to every element

# 7) Elementwise math vs matrix operations
a = np.array([1,2,3])
b = np.array([4,5,6])
a + b          # elementwise add -> [5,7,9]
a * b          # elementwise multiply -> [4,10,18]
A = np.arange(6).reshape(2,3)
B = np.arange(6).reshape(3,2)
A @ B          # matrix multiply -> shape (2,2)
np.dot(a, b)   # dot product (scalar)

# 8) Reductions and axis
M = np.arange(12).reshape(3,4)
M.sum()            # sum all
M.sum(axis=0)      # sum columns -> length 4
M.mean(axis=1)     # row means -> length 3
M.argmax()         # index of max in flattened array
M.argmax(axis=0)   # per-column argmax

# 9) Reshape, transpose, flatten
v = np.arange(6)
v.reshape(2,3)     # change shape (must keep same size)
v.reshape(-1, 2)   # -1 infers dimension
M.T                # transpose
v.ravel()          # view flattened
v.flatten()        # copy flattened

# 10) Stacking & splitting
a = np.array([1,2,3])
b = np.array([4,5,6])
np.concatenate([a,b])      # [1,2,3,4,5,6]
np.vstack([a,b])           # shape (2,3)
np.hstack([a.reshape(3,1), b.reshape(3,1)])  # shape (3,2)
np.split(np.arange(6), 3)  # split into 3 pieces

# 11) Useful utilities
np.unique(arr)                # unique values
np.sort(arr)                  # returns sorted copy
np.argsort(arr)               # indices that would sort
np.where(cond, x, y)          # choose x or y by condition
np.clip(arr, a_min, a_max)    # clamp values

# 12) Linear algebra & statistics
np.linalg.inv(A)              # inverse (A must be square)
np.linalg.eig(A)              # eigenvalues and eigenvectors
np.cov(data, rowvar=False)    # covariance
np.corrcoef(data, rowvar=False)

# 13) Random numbers (recommended API)
rng = np.random.default_rng(seed=42)
rng.integers(0, 10, size=(3,3))
rng.normal(loc=0, scale=1, size=1000)

# 14) I/O
np.save("arr.npy", arr)       # binary numpy format
a = np.load("arr.npy")
np.savez("many.npz", a=a, b=b)
np.savetxt("data.csv", arr, delimiter=",")   # text CSV (slower)
np.loadtxt("data.csv", delimiter=",")