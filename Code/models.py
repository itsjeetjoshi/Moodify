import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from nltk.corpus import stopwords
import string
import chardet
import joblib
import numpy as np
import nltk

nltk.download('stopwords')

# Detect encoding of the file
with open('sorted_data.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
print(f"Detected file encoding: {encoding}")

# Load the data
data = pd.read_csv('sorted_data.csv', encoding=encoding)

# Combine review_headline and review_body
data['combined_reviews'] = data['review_headline'].fillna('') + " " + data['review_body'].fillna('')

# Extract the combined reviews and sentiments
reviews = data['combined_reviews']
sentiments = data['Sentiment']  

# Preprocess the text data
def preprocess_text(text):
    if not isinstance(text, str):
        return ""  
    stop_words = set(stopwords.words('english'))
    text = text.lower() 
    text = ''.join([char for char in text if char not in string.punctuation]) 
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    return text

reviews_cleaned = reviews.apply(preprocess_text)

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews_cleaned)  
y = sentiments  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced", 
    classes=classes,
    y=y_train
)
class_weights_dict = dict(zip(classes, class_weights))
print("Class Weights:", class_weights_dict)

# Map weights to individual samples
sample_weights = y_train.map(class_weights_dict)

# Train the MultinomialNB model with class weights
model = MultinomialNB()
model.partial_fit(X_train, y_train, classes=classes, sample_weight=sample_weights)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative (-1)", "Neutral (0)", "Positive (1)"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("\nModel and vectorizer saved successfully.")