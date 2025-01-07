import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Step 1: Load the CSV file
df = pd.read_csv('C:\\Users\Admin\\Desktop\\Amazon-Product-Reviews - Amazon Product Review (1).csv')

# Step 2: Handle missing data
# Check for missing values
print(df.isnull().sum())

# Option 1: Drop rows with missing values
# df.dropna(inplace=True)

# Option 2: Impute missing values (for numerical columns)
# df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

# Option 3: Impute missing values for categorical columns (mode imputation)
# df['category_column'] = df['category_column'].fillna(df['category_column'].mode()[0])

# Step 3: Remove duplicate rows
df.drop_duplicates(inplace=True)

# Step 4: Convert data types (if necessary)
# For example, converting a column to datetime
# df['date_column'] = pd.to_datetime(df['date_column'])

# Step 5: Clean text data (if applicable)
def clean_text(text):
    text = str(text)
    # You can perform operations such as removing special characters, converting to lowercase, etc.
    text = text.lower()  # Convert to lowercase
    text = ''.join(e for e in text if e.isalnum() or e.isspace())  # Remove non-alphanumeric characters
    return text

# Apply text cleaning to a 'review_text' column
df['cleaned_review'] = df['review_body'].apply(clean_text)

# Step 6: Encoding categorical variables (if necessary)
# Example: Encoding a categorical column (e.g., 'category' column)
# Label Encoding
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['product_category'])

# Or One-hot encoding (for non-ordinal categories)
# df = pd.get_dummies(df, columns=['category_column'])

# Step 7: Feature extraction for text columns (if applicable)
# Using TF-IDF Vectorizer for text data
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # you can adjust max_features as needed
X_text = vectorizer.fit_transform(df['cleaned_review'])

# Step 8: Feature scaling for numerical data (if necessary)
# Example: Standardizing numerical columns
#scaler = StandardScaler()
#df[['numerical_column1', 'numerical_column2']] = scaler.fit_transform(df[['numerical_column1', 'numerical_column2']])

# Step 9: Split the data into features (X) and target (y)
# Assuming the target column is 'sentiment' (positive/negative/neutral)
#X = df.drop(columns=['sentiment'])  # Remove target column from features
#y = df['sentiment']  # Target column (sentiment)

# Step 10: Train-test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now your data is ready to be used for training a machine learning model!

# Optional: Save the cleaned dataframe to a new CSV for future use
df.to_csv('cleaned_data.csv', index=False)
