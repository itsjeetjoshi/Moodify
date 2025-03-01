import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

X = joblib.load("X_features.pkl")
y = joblib.load("Y_features.pkl")

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


joblib.dump(model, 'sentiment_model_with_kmeans.joblib')