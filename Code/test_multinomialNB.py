import joblib
import nltk
import string
from nltk.corpus import stopwords
from scipy.sparse import hstack

nltk.download('stopwords')

# Load the trained model, vectorizer, and K-Means clusterer
loaded_model = joblib.load('sentiment_model_with_kmeans.joblib')
vectorizer = joblib.load('vectorizer_with_kmeans.joblib')
kmeans = joblib.load('kmeans_clusterer.joblib')  # Load K-Means model

# Sample new reviews for prediction
new_reviews = ["This product is okay", "Worst experience ever. Do not buy.", "Great product"]

# Preprocess new reviews the same way as training data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

new_reviews_cleaned = [preprocess_text(review) for review in new_reviews]

# Transform new reviews using the same vectorizer (DO NOT FIT AGAIN)
new_reviews_vectorized = vectorizer.transform(new_reviews_cleaned)

# Assign clusters to new reviews using K-Means
new_clusters = kmeans.predict(new_reviews_vectorized)

# Append the cluster labels to the feature matrix
new_clusters = new_clusters.reshape(-1, 1)  # Reshape to match dimensions
new_reviews_vectorized = hstack((new_reviews_vectorized, new_clusters))  # Combine features

# Predict sentiment
predictions = loaded_model.predict(new_reviews_vectorized)

print("Predictions:", predictions)