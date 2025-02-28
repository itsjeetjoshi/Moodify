import pandas as pd
import numpy as np
import string
import joblib
import chardet
import nltk
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from scipy.sparse import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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
X = hstack((X, cluster_df.values))   # Combine features

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights))

# Map weights to individual samples
sample_weights = y_train.map(class_weights_dict)

# Train MultinomialNB model with class weights
model = MultinomialNB()
model.fit(X_train, y_train, sample_weight=sample_weights)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative (-1)", "Neutral (0)", "Positive (1)"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#FeedForward Neural Network

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train += 1
y_test += 1

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

model.save("sentiment_model.h5")

# Save model and vectorizer
joblib.dump(model, 'sentiment_model_with_kmeans.joblib')
joblib.dump(vectorizer, 'vectorizer_with_kmeans.joblib')
joblib.dump(kmeans, 'kmeans_clusterer.joblib')

print("\nModel, vectorizer, and K-Means clusterer saved successfully.")