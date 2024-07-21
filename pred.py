# -*- coding: utf-8 -*-
"""Copy of CustomRFClassifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BQJzAtOw5Kbt9Ef1mMwNDLWi8nVUM7SS
"""

import numpy as np
import pandas as pd
from collections import Counter
from math import log
import re

def equal_train_test_split_df(data, test_size=0.2):
    cities = ["Dubai", "New York City", "Paris", "Rio de Janeiro"]

    train_indices = []
    test_indices = []

    for city in cities:
        city_data = data[data['Label'] == city]
        shuffled_indices = np.random.permutation(len(city_data))
        num_test_samples = int(len(shuffled_indices) * test_size)
        test_indices.extend(city_data.index[shuffled_indices[:num_test_samples]].tolist())
        train_indices.extend(city_data.index[shuffled_indices[num_test_samples:]].tolist())

    train_data = data.loc[train_indices]
    test_data = data.loc[test_indices]

    return train_data, test_data

def extract_rankings(text):
    rankings = dict(item.split('=>') for item in text.split(','))
    return {k: int(v) if v else -1 for k, v in rankings.items()}

def adjust_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data

def clean_text(text):
      text = ''.join([c if ord(c) < 128 else '' for c in text])
      text = re.sub(r'[^\w\s]', '', text).lower()
      text = re.sub(r'\s+', ' ', text).strip()
      return text

class CustomPipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_tf = X
        # Everything before final step is a transformer.
        for _, process in self.steps[:-1]:
             X_tf = process.fit_transform(X_tf, y)
        # Last is always a model
        self.steps[-1][1].fit(X_tf, y)
        return self

    def predict(self, X):
        X_tf = X
        for _, process in self.steps[:-1]:
            X_tf = process.transform(X_tf)
        return self.steps[-1][1].predict(X_tf)

class BaseTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError(
            "Each transformer must implement its own transform method.")

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class TextCleaner(BaseTransformer):
    def __init__(self, column=None):
        self.column = column

    def fit(self, X, y=None):
        # Nothing to learn from the data
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].fillna('').apply(clean_text)
        return X

class NumericInputConverter(BaseTransformer):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            # Convert the column to string to ensure the .str accessor works
            X[col] = X[col].astype(str).str.replace(',', '').astype(float)
        return X

class CategoricalEncoder(BaseTransformer):
    def __init__(self, column=None, categories=None):
        self.column = column
        self.categories = categories

    def fit(self, X, y=None):
        # No fitting needed as transformation is rule-based
        return self

    def transform(self, X):
        X = X.copy()
        for category in self.categories:
            X[category] = X[self.column].apply(lambda x: 1 if isinstance(x, str) and category in x.split(',') else 0)
        X.drop([self.column], axis=1, inplace=True)
        return X

class OutlierAdjuster(BaseTransformer):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        # No fitting process needed as adjustment is rule-based
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X[column] = X[column].clip(lower=lower_bound, upper=upper_bound)
        return X

class MedianImputer(BaseTransformer):
    def __init__(self, columns=None):
        self.columns = columns
        self.med_vals = None

    def fit(self, X, y=None):
        self.med_vals = {col: X[col].median() for col in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col].fillna(self.med_vals[col], inplace=True)
        return X

class MeanImputer(BaseTransformer):
    def __init__(self, columns=None):
        self.columns = columns
        self.mean_vals = None

    def fit(self, X, y=None):
        self.mean_vals = {col: X[col].mean() for col in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col].fillna(self.mean_vals[col], inplace=True)
        return X

class RankingsExtractor(BaseTransformer):
    def __init__(self, column=None):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        rankings = X[self.column].apply(extract_rankings)
        rankings_df = pd.json_normalize(rankings)
        rankings_df.index = X.index
        return pd.concat([X.drop([self.column], axis=1), rankings_df], axis=1)

class CustomTfidfVectorizer(BaseTransformer):
    def __init__(self, column=None, stopwords_path='stopwords.txt', max_df=0.5, min_df=8):
        self.column = column
        self.stopwords_path = stopwords_path
        self.max_df = max_df
        self.min_df = min_df
        self.vocab_idf = None
        self.num_docs = 0
        self.word_index = None

    def fit(self, X, y=None):
        with open(self.stopwords_path, 'r') as f:
            stopwords = f.read().splitlines()

        doc_freq = Counter()
        for text in X[self.column].fillna('').apply(clean_text):
            words = set(word for word in text.split() if word not in stopwords)
            doc_freq.update(words)

        self.num_docs = len(X)
        max_df_abs = self.max_df if isinstance(self.max_df, int) else int(self.max_df * self.num_docs)
        min_df_abs = self.min_df if isinstance(self.min_df, int) else int(self.min_df * self.num_docs)
        self.vocab_idf = {word: log((1 + self.num_docs) / (1 + doc_freq[word])) + 1 for word in doc_freq if min_df_abs <= doc_freq[word] <= max_df_abs}
        self.word_index = {word: i for i, word in enumerate(self.vocab_idf)}

        return self

    def transform(self, X):
        X = X.copy()
        num_docs = len(X)
        tf_idf_matrix = np.zeros((num_docs, len(self.vocab_idf)))

        for doc_idx, text in enumerate(X[self.column].fillna('').apply(clean_text)):
            words = text.split()
            term_freq = Counter(word for word in words if word in self.vocab_idf)
            for word, tf in term_freq.items():
                if word in self.word_index:
                    idf = self.vocab_idf[word]
                    tf_idf_matrix[doc_idx, self.word_index[word]] = tf * idf

        norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tf_idf_matrix /= norms

        tf_idf_df = pd.DataFrame(tf_idf_matrix, columns=list(self.vocab_idf), index=X.index)
        return pd.concat([X.drop([self.column], axis=1), tf_idf_df], axis=1)

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

def create_pipeline():
    pipeline = CustomPipeline(steps=[('median_imputer', MedianImputer(columns=['Q1', 'Q2', 'Q3', 'Q4'])),
    ('numeric_converter', NumericInputConverter(columns=['Q7', 'Q8', 'Q9'])),
    ('outlier_adjuster', OutlierAdjuster(columns=['Q7', 'Q8', 'Q9'])),
    ('mean_imputer', MeanImputer(columns=['Q7', 'Q8', 'Q9'])),
    ('categorical_encoder', CategoricalEncoder(column='Q5', categories=["Siblings", "Co-worker", "Partner", "Friends"])),
    ('rankings_extractor', RankingsExtractor(column='Q6')),
    ('text_cleaner', TextCleaner(column='Q10')),
    ('tfidf_vectorizer', CustomTfidfVectorizer(column='Q10')),
    ('random_forest', RandomForest(num_trees=100, max_depth=8, max_features=8))])
    return pipeline

def trained_model(file):
    data = pd.read_csv(file)
    np.random.seed(42)
    X = data.drop(['id', 'Label'], axis=1)
    y = data['Label'].reset_index(drop=True)
    city_mapping = {'Dubai': 0, 'New York City': 1, 'Paris': 2, 'Rio de Janeiro': 3}
    y_encoded = np.array([city_mapping[city] for city in y])
    pipeline = create_pipeline()
    return pipeline.fit(X, y_encoded)

def predict_all(filename):
    data = pd.read_csv(filename)
    data = data.drop(['id'], axis=1)
    model = trained_model('clean_dataset.csv')
    numeric_predictions = model.predict(data)
    reverse_city_mapping = {0: 'Dubai', 1: 'New York City', 2: 'Paris', 3: 'Rio de Janeiro'}
    predictions = [reverse_city_mapping[num] for num in numeric_predictions]
    return predictions

if __name__ == "__main__":
    model = trained_model('clean_dataset.csv')
    predictions = predict_all('sample_dataset.csv')
    print(predictions)