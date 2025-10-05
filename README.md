# Text Classification with Logistic Regression for Sentiment Analysis

## Project Overview
This project involves analyzing customer reviews to classify them as **positive** or **negative** using **Logistic Regression**. The workflow includes text preprocessing, feature extraction, training a model, making predictions, and evaluating its performance.

## Dataset  
The dataset consists of customer reviews labeled with sentiment scores:  
- **Review**: The text of the customer’s review  
- **Sentiment**: The target variable (1 = Positive, 0 = Negative)  

**Dataset file**: `customer_reviews_sentiment.csv`

## Requirements  
Install the necessary dependencies before running the code:  
```bash
pip install pandas numpy scikit-learn nltk
```

## Workflow  

### 1. Data Preprocessing  
- Load the dataset  
- Handle missing values  
- Convert text to lowercase  
- Remove special characters and punctuation  
- Remove stop words (using NLTK or SpaCy)  
- Tokenize the text  

### 2. Feature Extraction  
- Convert text into numerical features using:  
  - **Bag of Words (BoW)** or  
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**  
- Split the dataset into training (80%) and testing (20%) sets  

### 3. Train a Logistic Regression Model  
- Train a **Logistic Regression classifier** on extracted features  
- Tune hyperparameters (experiment with regularization parameter **C**)  

### 4. Make Predictions  
- Predict the sentiment for the following reviews:  
  - `"This product is amazing! I love it."`  
  - `"It broke after one use, completely disappointed."`  

### 5. Model Evaluation  
- Compute **accuracy** on the test dataset  
- Generate **confusion matrix** and **classification report** (precision, recall, F1-score)  

### 6. Model Improvements  
- Experiment with other classifiers (e.g., **Naive Bayes, SVM**)  
- Compare their performance with Logistic Regression  

## Deliverables  
1. **Preprocessed Dataset**: Cleaned text data  
2. **Feature-Engineered Dataset**: Extracted numerical features  
3. **Trained Model**: Logistic Regression with optimized hyperparameters  
4. **Model Evaluation**: Accuracy, confusion matrix, classification report  
5. **Sample Predictions**: Results for provided test cases  
6. **Model Comparison**: Performance of alternative classifiers  

## How to Run the Code  
1. Ensure the dataset is available as `customer_reviews_sentiment.csv`  
2. Run the preprocessing and feature extraction scripts  
3. Train the Logistic Regression model  
4. Evaluate performance and compare models  

## Author  
**Pedahel Emmanuel Kojo**  
Senior Software Engineer, Machine Learning Engineer at CSP
