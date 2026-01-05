import numpy as np
from collections import Counter


# --------------------------
# Utility functions
# --------------------------

def entropy(y):
    y = np.asarray(y).astype(int)
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])


def information_gain(y, y_left, y_right):
    H_parent = entropy(y)

    w_left = len(y_left) / len(y)
    w_right = len(y_right) / len(y)

    return H_parent - (w_left * entropy(y_left) + w_right * entropy(y_right))


# --------------------------
# Decision Tree Classifier
# --------------------------

class DecisionTree:
    def __init__(self, max_depth=10, min_samples=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.tree_ = None

    # ---- core helpers ----
    def _best_split(self, X, y, feature_indices):
        best_ig = -1.0
        best_feature = None
        best_thresh = None

        for feature in feature_indices:
            values = np.sort(np.unique(X[:, feature]))
            if len(values) < 2:
                continue

            for i in range(len(values) - 1):
                thresh = (values[i] + values[i + 1]) / 2  #we choose thr mid point where the classes are separated

                left_mask = X[:, feature] <= thresh
                right_mask = ~left_mask #flips boolean feature..the feature of left mask gets inverted onto the right mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                ig = information_gain(y, y[left_mask], y[right_mask])

                if ig > best_ig:
                    best_ig = ig
                    best_feature = feature
                    best_thresh = thresh

        return best_feature, best_thresh, best_ig

    def _build(self, X, y, depth):
        # stopping rules
        if len(set(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples:
            return Counter(y).most_common(1)[0][0]

        n_features_total = X.shape[1]

        # choose subset of features (Random Forest style if max_features set)
        n_features = (
            self.max_features
            if self.max_features is not None
            else n_features_total
        )

        feature_indices = np.random.choice(
            n_features_total, n_features, replace=False
        )

        feature, thresh, best_ig = self._best_split(X, y, feature_indices)

        if feature is None or best_ig <= 0:
            return Counter(y).most_common(1)[0][0]

        left_mask = X[:, feature] <= thresh
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return Counter(y).most_common(1)[0][0]

        left_tree = self._build(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build(X[right_mask], y[right_mask], depth + 1)

        return (feature, thresh, left_tree, right_tree)

    # ---- public API ----
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        # leaf
        if not isinstance(node, tuple):
            return node

        feature, thresh, left, right = node

        if x[feature] <= thresh:
            return self._predict_one(x, left)
        else:
            return self._predict_one(x, right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(row, self.tree_) for row in X])
