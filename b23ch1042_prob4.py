import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. DATA COLLECTION & PREPROCESSING
# ==========================================
def get_data():
    print("Loading 20 Newsgroups dataset...")
    # Define the categories we want
    categories = [
        'rec.sport.baseball', 'rec.sport.hockey',  # Sports
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc' # Politics
    ]
    
    # Load data (removing headers/footers prevents overfitting to metadata)
    dataset = fetch_20newsgroups(subset='all', categories=categories, 
                                 remove=('headers', 'footers', 'quotes'))
    
    # Create Binary Labels: 1 = Sports, 0 = Politics
    # We check if the category name contains 'sport'
    y = np.array([1 if 'sport' in dataset.target_names[t] else 0 for t in dataset.target])
    X = dataset.data
    
    return X, y

# ==========================================
# 2. MAIN PIPELINE
# ==========================================
def main():
    # A. Load Data
    X_raw, y = get_data()
    print(f"Data Loaded: {len(X_raw)} documents.")
    print(f"Class Distribution: {sum(y)} Sports, {len(y)-sum(y)} Politics")

    # B. Feature Representation (TF-IDF + N-Grams)
    # We use (1,2) grams to capture phrases like "home run" or "foreign policy"
    print("Vectorizing text (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    X_vectorized = tfidf.fit_transform(X_raw)
    
    # C. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # D. Define Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine (SVM)": SVC(kernel='linear')
    }

    # E. Train and Evaluate
    results = {}
    
    print("\n" + "="*40)
    print("MODEL PERFORMANCE REPORT")
    print("="*40)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Politics', 'Sports'])
        cm = confusion_matrix(y_test, predictions)
        
        results[name] = acc
        
        print(f"--- {name} Results ---")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)
        print(f"Confusion Matrix:\n{cm}")

    # F. Final Comparison
    print("\n" + "="*40)
    print("FINAL ACCURACY RANKING")
    print("="*40)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(sorted_results, 1):
        print(f"{rank}. {name}: {score*100:.2f}%")

if __name__ == "__main__":
    main()