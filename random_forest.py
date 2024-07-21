import numpy as np
import pandas as pd

class RandomForest:
    def __init__(self, num_trees=50, max_depth=8, max_features=8):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
 
    class _DecisionTree:
        def __init__(self, max_depth, max_features):
            self.tree = None
            self.max_depth = max_depth
            self.max_features = max_features
 
        def _entropy(self, y):
            unique, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return -np.sum(probabilities * np.log2(probabilities))
 
        def _calculate_max_features(self, total_features):
            if isinstance(self.max_features, str):
                if self.max_features == 'sqrt':
                    return int(np.sqrt(total_features))
                elif self.max_features == 'log2':
                    return int(np.log2(total_features))
            elif isinstance(self.max_features, float):
                return int(total_features * self.max_features)
            elif self.max_features is None:
                return total_features
            else:  # Assume int
                return min(total_features, self.max_features)
 
        def _best_split(self, X, y):
            n_features = X.shape[1]
            max_feats = self._calculate_max_features(n_features)
 
            feature_indices = np.random.choice(range(n_features), max_feats, replace=False) if max_feats < n_features else range(n_features)
 
            best_feature_idx, best_threshold, best_entropy = None, None, float('inf')
 
            for feature_idx in feature_indices:
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    left_mask = X[:, feature_idx] <= threshold
                    right_mask = ~left_mask
 
                    if not np.any(left_mask) or not np.any(right_mask):
                        continue
 
                    left_entropy = self._entropy(y[left_mask])
                    right_entropy = self._entropy(y[right_mask])
                    entropy = (np.sum(left_mask) * left_entropy + np.sum(right_mask) * right_entropy) / len(y)
 
                    if entropy < best_entropy:
                        best_feature_idx, best_threshold, best_entropy = feature_idx, threshold, entropy
 
            return best_feature_idx, best_threshold
 
        def fit(self, X, y, depth=0):
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
 
            if len(np.unique(y)) == 1 or len(y) <= 1 or depth >= self.max_depth:
                return np.bincount(y).argmax()
 
            best_feature_idx, best_threshold = self._best_split(X, y)
            if best_feature_idx is None:  # No valid split was found
                return np.bincount(y).argmax()
 
            left_mask = X[:, best_feature_idx] <= best_threshold
            right_mask = ~left_mask
            left_subtree = self.fit(X[left_mask], y[left_mask], depth + 1)
            right_subtree = self.fit(X[right_mask], y[right_mask], depth + 1)
 
            self.tree = ((best_feature_idx, best_threshold), left_subtree, right_subtree)
            return self.tree
 
        def predict(self, X):
            if self.tree is None:
                raise ValueError("This Decision Tree is not fitted yet.")
 
            if isinstance(X, pd.DataFrame):
                X = X.values
 
            predictions = np.array([self._predict_one(sample, self.tree) for sample in X])
            return predictions
 
        def _predict_one(self, x, tree):
            if not isinstance(tree, tuple):
                return tree
            else:
                feature_idx, threshold = tree[0]
                if x[feature_idx] <= threshold:
                    return self._predict_one(x, tree[1])
                else:
                    return self._predict_one(x, tree[2])
 
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            dt = self._DecisionTree(max_depth=self.max_depth, max_features=self.max_features)
            X_sampled, y_sampled = self._bootstrap_sample(X, y)
            dt.tree = dt.fit(X_sampled, y_sampled)
            self.trees.append(dt)
 
    def predict(self, X):
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        if all_predictions.shape[0] > 1:
            final_predictions = np.array([np.bincount(predictions.astype(int)).argmax() for predictions in all_predictions.T])
        else:
            final_predictions = all_predictions[0]
        return final_predictions
 
    def _bootstrap_sample(self, X, y):
        n_samples = len(y)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if isinstance(X, pd.DataFrame):
            X_sampled = X.iloc[indices].reset_index(drop=True)
        else:
            X_sampled = X[indices]
 
        if isinstance(y, pd.Series):
            y_sampled = y.iloc[indices].reset_index(drop=True)
        else:
            y_sampled = y[indices]
 
        return X_sampled, y_sampled
