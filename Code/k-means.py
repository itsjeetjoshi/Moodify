import pandas as pd
import string
import joblib
import chardet
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from scipy.sparse import hstack

nltk.download('stopwords')

# Detect file encoding
with open('sorted_data.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
print(f"Detected file encoding: {encoding}")

# Load the dataset
data = pd.read_csv('sorted_data.csv', encoding=encoding)

# Combine review_headline and review_body
data['combined_reviews'] = data['review_headline'].fillna('') + " " + data['review_body'].fillna('')

# Extract the combined reviews and sentiments
reviews = data['combined_reviews']
sentiments = data['Sentiment']

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

reviews_cleaned = reviews.apply(preprocess_text)

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews_cleaned)
y = sentiments

# Apply K-Means Clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Convert clusters to DataFrame and append as a feature
cluster_df = pd.DataFrame(cluster_labels, columns=['Cluster'])
print(cluster_df)
X = hstack((X, cluster_df.values)) 
print(X)  # Combine features

joblib.dump(X, "X_features.pkl")
joblib.dump(y, "Y_features.pkl")

# Save model and vectorizer
joblib.dump(vectorizer, 'vectorizer_with_kmeans.joblib')
joblib.dump(kmeans, 'kmeans_clusterer.joblib')

print("\nModel, vectorizer, and K-Means clusterer saved successfully.")