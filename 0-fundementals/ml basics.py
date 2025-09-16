# Data loading & quick inspection
import pandas as pd
import numpy as np

# load
df = pd.read_csv("data.csv")      # or pd.read_parquet, pd.read_json, etc.

# quick look
df.head()
df.info()
df.describe(include='all')
df.isna().sum()

# Simple preprocessing helpers
# drop or impute missing values
df = df.dropna(subset=['target'])              # must-have rows
df['age'] = df['age'].fillna(df['age'].median())

# convert datetime
df['dt'] = pd.to_datetime(df['dt'])
df['month'] = df['dt'].dt.month

# label encode low-cardinality categorical (map)
df['gender_code'] = df['gender'].map({'M':0, 'F':1}).astype('Int64')

# one-hot encode (pandas)
df = pd.get_dummies(df, columns=['city'], drop_first=True)

# feature / target split
target = 'price'
X_df = df.drop(columns=[target])
y = df[target].values.reshape(-1, 1)   # (n,1)

# Convert between pandas and NumPy
# Keep feature names
feature_names = X_df.columns.tolist()
X = X_df.values    # numpy array shape (n_samples, n_features)

# Train / test split (numpy)
def train_test_split(X, y, test_ratio=0.2, seed=None):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1 - test_ratio))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)

# Scaling (standardization)
class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression

# Normal equation (closed-form) and gradient descent versions:

def linear_regression_normal_eq(X, y, l2=0.0):
    # add bias column
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    # theta = (X^T X + lambda*I)^(-1) X^T y
    d = Xb.shape[1]
    I = np.eye(d)
    I[0,0] = 0.0  # don't regularize bias
    theta = np.linalg.pinv(Xb.T @ Xb + l2 * I) @ (Xb.T @ y)
    return theta  # shape (d,1)

def predict_linear(X, theta):
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    return Xb @ theta

# Gradient descent (batch)
def linear_regression_gd(X, y, lr=1e-3, epochs=1000, l2=0.0):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    d = Xb.shape[1]
    theta = np.zeros((d,1))
    n = Xb.shape[0]
    for _ in range(epochs):
        grad = (Xb.T @ (Xb @ theta - y)) / n + l2 * np.r_[ [[0]], theta[1:] ]
        theta -= lr * grad
    return theta

# Logistic Regression (binary) — vectorized
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_gd(X, y, lr=1e-2, epochs=2000, l2=0.0):
    # y expected shape (n,1) with 0/1 labels
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    n, d = Xb.shape
    w = np.zeros((d,1))
    for _ in range(epochs):
        z = Xb @ w
        p = sigmoid(z)
        grad = (Xb.T @ (p - y)) / n + l2 * np.r_[ [[0]], w[1:] ]
        w -= lr * grad
    return w

def predict_proba_logistic(X, w):
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    return sigmoid(Xb @ w)

def predict_logistic(X, w, thresh=0.5):
    return (predict_proba_logistic(X, w) >= thresh).astype(int)

# Evaluation metrics
def mse(y_true, y_pred): return np.mean((y_true - y_pred)**2)
def rmse(y_true, y_pred): return np.sqrt(mse(y_true, y_pred))
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - ss_res / ss_tot

# classification
def accuracy(y_true, y_pred): return np.mean(y_true.ravel() == y_pred.ravel())
def precision_recall_f1(y_true, y_pred):
    y_true = y_true.ravel(); y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

# PCA via SVD (dimensionality reduction)
def pca_svd(X, k):
    # X: centered data (n, d)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:k]              # shape (k, d)
    projected = X @ components.T     # shape (n, k)
    return projected, components, S

# example: center data then project
X_centered = X_train_scaled - X_train_scaled.mean(axis=0)
X_pca, comps, S = pca_svd(X_centered, k=2)

# Simple k-fold cross-validation (regression example)
def k_fold_indices(n, k, seed=None):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)

def cross_val_score_linear(X, y, k=5):
    n = X.shape[0]
    folds = k_fold_indices(n, k, seed=1)
    scores = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([f for j,f in enumerate(folds) if j!=i])
        theta = linear_regression_normal_eq(X[train_idx], y[train_idx])
        y_pred = predict_linear(X[val_idx], theta)
        scores.append(rmse(y[val_idx], y_pred))
    return np.mean(scores), np.std(scores)

# Small end-to-end example (regression)
# assume df already preprocessed, X, y defined
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

theta = linear_regression_normal_eq(X_train_s, y_train, l2=1e-3)
y_pred = predict_linear(X_test_s, theta)
print("RMSE:", rmse(y_test, y_pred), "R2:", r2(y_test, y_pred))