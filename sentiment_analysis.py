import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Define custom NLTK data path
nltk_data_path = os.path.expanduser('/mnt/c/python_projects/.nltk_data')
nltk.data.path.append(nltk_data_path)

# Ensure necessary NLTK datasets are downloaded
required_nltk_resources = ['stopwords', 'punkt']
for resource in required_nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

# Load stopwords
stop_words = set(stopwords.words('english'))

# File path
file_path = 'customer_reviews_sentiment.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the path.")

# Load dataset
df = pd.read_csv(file_path)

# Check and handle missing values
df.dropna(subset=['Review', 'Sentiment'], inplace=True)

# Convert reviews to lowercase
df['Review'] = df['Review'].str.lower()

# Remove special characters, digits, and punctuation
df['Review'] = df['Review'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

# Tokenize and remove stopwords
df['Review'] = df['Review'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

# Split dataset into features and target
X = df['Review']
y = df['Sentiment']

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Stratified train-test split to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution
class_counts = y_train.value_counts()
min_samples = class_counts.min()

# Handle class imbalance using SMOTE or fallback to RandomOverSampler
if min_samples >= 6:
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
else:
    print("Not enough samples for SMOTE, using RandomOverSampler instead.")
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Hyperparameter tuning for Logistic Regression
param_grid = {'C': [0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000), param_grid, cv=5)
grid.fit(X_train_resampled, y_train_resampled)
log_reg_best = grid.best_estimator_

# Predict on test set
y_pred = log_reg_best.predict(X_test)

# Evaluate model performance
print("Logistic Regression Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Make predictions on new reviews
sample_reviews = ["This product is amazing! I love it.", "It broke after one use, completely disappointed."]
sample_transformed = vectorizer.transform(sample_reviews)
print("Sample Predictions:", log_reg_best.predict(sample_transformed))

# Alternative Model: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_resampled, y_train_resampled)
y_pred_nb = nb_model.predict(X_test)

# Compare with Naive Bayes
print("Naive Bayes Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))
print("Classification Report:")
print(classification_report(y_test, y_pred_nb, zero_division=1))
