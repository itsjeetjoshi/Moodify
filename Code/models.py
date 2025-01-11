import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
import chardet
import joblib

# Detect encoding
with open('sorted_data.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
print(f"Detected file encoding: {encoding}")

data = pd.read_csv('sorted_data.csv', encoding=encoding)

reviews = data['review_body']
sentiments = data['Sentiment']  

def preprocess_text(text):
    if not isinstance(text, str):
        return ""  
    stop_words = set(stopwords.words('english'))
    text = text.lower() 
    text = ''.join([char for char in text if char not in string.punctuation]) 
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    return text

reviews_cleaned = reviews.apply(preprocess_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews_cleaned)  
y = sentiments  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative (-1)", "Neutral (0)", "Positive (1)"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("\nModel and vectorizer saved successfully.")