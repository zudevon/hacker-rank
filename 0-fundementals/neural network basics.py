# Neural network fundamentals script (pure NumPy)
# This will run a small example: train a 1-hidden-layer neural network on a toy binary classification task.
# It prints training progress and plots training loss & accuracy.
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Utilities & synthetic data -------------------------
def make_blobs(n_samples=400, centers=2, dim=2, spread=1.0, seed=1):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for c in range(centers):
        center = rng.normal(loc=(c * 3.0), scale=0.5, size=(dim,))
        pts = rng.normal(loc=center, scale=spread, size=(n_samples // centers, dim))
        X.append(pts)
        y.append(np.full((n_samples // centers,), c))
    X = np.vstack(X)
    y = np.concatenate(y)
    # shuffle
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]

# Activation functions and derivatives
def sigmoid(z):
    # stable sigmoid
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def tanh(z):
    return np.tanh(z)

def tanh_deriv(a):
    return 1 - a**2

# Loss functions
def binary_cross_entropy(y_true, y_prob):
    # y_true shape (n,)
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

# ------------------------- Simple Feedforward Neural Network -------------------------
class SimpleNN:
    def __init__(self, n_inputs, n_hidden=16, activation='relu', seed=1):
        rng = np.random.default_rng(seed)
        # He / Xavier initialization depending on activation
        if activation == 'relu':
            w1_scale = np.sqrt(2.0 / n_inputs)
        else:
            w1_scale = np.sqrt(1.0 / n_inputs)
        self.W1 = rng.normal(scale=w1_scale, size=(n_inputs, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(scale=np.sqrt(1.0 / n_hidden), size=(n_hidden, 1))
        self.b2 = np.zeros((1, 1))
        self.activation = activation

    def _activate(self, z):
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'tanh':
            return tanh(z)
        else:
            return sigmoid(z)

    def _activate_deriv(self, pre_or_act):
        if self.activation == 'relu':
            return relu_deriv(pre_or_act)  # pass pre-activation z
        elif self.activation == 'tanh':
            return tanh_deriv(pre_or_act)  # pass activation a
        else:
            return sigmoid_deriv(pre_or_act)  # pass activation a

    def forward(self, X):
        # X: (n, d)
        self.Z1 = X @ self.W1 + self.b1      # (n, hidden)
        self.A1 = self._activate(self.Z1)   # (n, hidden)
        self.Z2 = self.A1 @ self.W2 + self.b2  # (n, 1)
        self.A2 = sigmoid(self.Z2).reshape(-1) # output probabilities shape (n,)
        return self.A2

    def backward(self, X, y_true, lr=0.01, l2=0.0):
        # y_true shape (n,)
        n = X.shape[0]
        y_pred = self.A2.reshape(-1,1)  # (n,1)
        y_true_col = y_true.reshape(-1,1)
        # dLoss/dZ2 for BCE with sigmoid output: (y_pred - y_true) / n
        dZ2 = (y_pred - y_true_col) / n  # (n,1)
        dW2 = self.A1.T @ dZ2 + l2 * self.W2  # (hidden,1)
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1,1)

        dA1 = dZ2 @ self.W2.T  # (n, hidden)
        # derivative for hidden layer
        if self.activation == 'relu':
            dZ1 = dA1 * relu_deriv(self.Z1)
        elif self.activation == 'tanh':
            dZ1 = dA1 * tanh_deriv(self.A1)
        else:
            dZ1 = dA1 * sigmoid_deriv(self.A1)
        dW1 = X.T @ dZ1 + l2 * self.W1  # (n_inputs, hidden)
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, hidden)

        # gradient descent update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, thresh=0.5):
        return (self.predict_proba(X) >= thresh).astype(int)

# ------------------------- Training loop / example -------------------------
def train_demo():
    # Data
    X, y = make_blobs(n_samples=600, centers=2, dim=2, spread=0.9, seed=2)
    # normalize features (simple scaling)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / (X_std + 1e-8)

    # split train/test
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # model
    model = SimpleNN(n_inputs=X.shape[1], n_hidden=16, activation='relu', seed=42)

    # training hyperparams
    epochs = 200
    lr = 0.5
    batch_size = 32
    l2 = 1e-4

    history = {'loss': [], 'acc': []}

    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        # mini-batch SGD
        idx = rng.permutation(X_train.shape[0])
        X_shuffled = X_train[idx]; y_shuffled = y_train[idx]
        for start in range(0, X_train.shape[0], batch_size):
            end = start + batch_size
            xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]
            model.forward(xb)
            model.backward(xb, yb, lr=lr, l2=l2)

        # evaluate on training set each epoch
        y_prob = model.predict_proba(X_train)
        loss = binary_cross_entropy(y_train, y_prob)
        y_pred = (y_prob >= 0.5).astype(int)
        acc = np.mean(y_pred == y_train)
        history['loss'].append(loss)
        history['acc'].append(acc)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:03d}/{epochs}  loss={loss:.4f}  acc={acc:.4f}")

    # final test evaluation
    y_test_prob = model.predict_proba(X_test)
    test_loss = binary_cross_entropy(y_test, y_test_prob)
    test_acc = np.mean((y_test_prob >= 0.5).astype(int) == y_test)
    print(f"\nFinal test loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    # Plot training loss
    plt.figure()
    plt.plot(history['loss'])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Plot training accuracy
    plt.figure()
    plt.plot(history['acc'])
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    # Decision boundary visualization (optional)
    xx = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200)
    yy = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200)
    grid_xx, grid_yy = np.meshgrid(xx, yy)
    grid = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])
    grid_std = (grid - X_mean) / (X_std + 1e-8)
    probs = model.predict_proba(grid_std).reshape(grid_xx.shape)

    plt.figure()
    plt.contourf(grid_xx, grid_yy, probs, levels=20)
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, marker='o', label='train')
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker='x', label='test')
    plt.title("Decision Boundary (probability)")
    plt.legend()
    plt.show()

train_demo()
