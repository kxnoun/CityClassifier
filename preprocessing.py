import numpy as np
import pandas as pd
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