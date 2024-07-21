import pandas as pd
import numpy as np
from pipeline import CustomPipeline, MedianImputer, NumericInputConverter, OutlierAdjuster, MeanImputer, CategoricalEncoder, RankingsExtractor, TextCleaner, CustomTfidfVectorizer
from random_forest import RandomForest

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