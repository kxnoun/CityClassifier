### README.md

# 🌆 City Predictor: UofT Student Survey Analysis 🌇

Welcome to the **City Predictor** project! This repository contains the code and data for a machine learning model that predicts cities based on survey responses from students at the University of Toronto. The model guesses one of four cities: Rio de Janeiro, Dubai, New York City, or Paris, based on answers to 10 questions.

## 📊 Project Overview

In this project, we:
- Collected survey data with responses to 10 questions describing a city.
- Explored and cleaned the data, addressing missing values and outliers.
- Tested multiple machine learning algorithms.
- Chose the Random Forest model for its high accuracy.
- Built a custom model from scratch, including preprocessing pipelines and hyperparameter tuning.

Feel free to explore the repository to see how different models were tested and to understand the custom implementation of our chosen model.

## 🚀 Getting Started

To run the code, you'll need Python 3 installed on your machine, along with `pandas` and `numpy`. The cool part of this project is that we did not use any prebuilt models or libraries like `scikit-learn` for the final implementation!

### Prerequisites

Make sure you have the following packages installed:
```bash
pip install pandas numpy
```

### Running the Model

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kxnoun/city-predictor.git
   cd city-predictor
   ```

2. **Explore the code:**
   - `preprocessing.py`: Functions for data preprocessing.
   - `pipeline.py`: Custom pipeline and transformers.
   - `random_forest.py`: Custom Random Forest implementation.
   - `alternate_models/`: Directory containing different models tried during the project.
   - `main.py`: Main script for training and predicting.

3. **Run the main script:**
   ```bash
   python main.py
   ```

This will train the model on the provided dataset and make predictions based on the survey responses.

## 📂 Project Structure

```
city-predictor/
│
├── preprocessing.py
├── pipeline.py
├── random_forest.py
├── main.py
├── alternate_models/
│   ├── kNearestNeighbour.ipynb
│   ├── LogisticRegression.ipynb
│   └── ...
├── clean_dataset.csv
└── stopwords.txt
```

## 📚 Data

The dataset contains survey responses with the following features:
- **Q1-Q4**: Likert scale questions (1 to 5)
- **Q5**: Categories of travel companions (binary: Siblings, Co-worker, Partner, Friends)
- **Q6**: Rankings of city aspects (1 to 6: Skyscrapers, Sport, Art and Music, Carnival, Cuisine, Economic)
- **Q7**: Average temperature
- **Q8**: Number of languages spoken
- **Q9**: Number of fashion styles
- **Q10**: Descriptive text about the city

## 🛠️ Custom Pipeline

The custom pipeline includes steps like:
- Imputing missing values.
- Converting numeric inputs.
- Adjusting outliers.
- Encoding categorical data.
- Extracting rankings.
- Cleaning text.
- Vectorizing text using TF-IDF.
- Training the Random Forest model.

## 📈 Model Performance

After trying various models, we found that the Random Forest model provided the highest accuracy on the test set. We fine-tuned the hyperparameters to achieve optimal performance.

## 🎉 Contributors

- **Adam**: Data Pre-processing, Model Exploration (Random Forest), Model Implementation, Hyper-parameter Fine-tuning, Cross-validation
- **Neha**: Data and Model Exploration (Logistic Regression, kNN), `pred.py` script Implementation, Report Writing
- **Katerina**: Data Exploration and Visualization
- **Manahil**: Model Exploration (Decision Tree), Custom RF Implementation for `pred.py`, Report Writing
