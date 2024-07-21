import pandas as pd
import numpy as np
from preprocessing import clean_text, extract_rankings, adjust_outliers
from collections import Counter
from math import log


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