import joblib
from nltk.corpus import stopwords
import string

# Load the saved model and vectorizer
loaded_model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Sample new review(s) for prediction
new_reviews = ["This product is okay", "Worst experience ever. Do not buy."]

# Preprocess new reviews the same way as training data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

new_reviews_cleaned = [preprocess_text(review) for review in new_reviews]

# Transform new reviews using the same vectorizer (Do NOT fit again)
new_reviews_vectorized = vectorizer.transform(new_reviews_cleaned)

# Predict sentiment
predictions = loaded_model.predict(new_reviews_vectorized)

print("Predictions:", predictions)