import joblib
from nltk.corpus import stopwords
import string

loaded_model = joblib.load('sentiment_model.joblib')
loaded_vectorizer = joblib.load('vectorizer.joblib')

def preprocess_text(text):
    if not isinstance(text, str):  # Check if the input is not a string
        return ""  
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Predict new reviews
new_reviews = ["Five Stars for this being the worst product"]
new_reviews_cleaned = [preprocess_text(review) for review in new_reviews]
new_reviews_vectorized = loaded_vectorizer.transform(new_reviews_cleaned)
predictions = loaded_model.predict(new_reviews_vectorized)

predicted_labels = []

# Map predictions back to sentiment labels if needed
for label in predictions:
    if label == 1:
        predicted_labels.append("Positive")
    elif label == 0:
        predicted_labels.append("Neutral")
    elif label == -1:
        predicted_labels.append("Negative")

print("Predictions:", predicted_labels)