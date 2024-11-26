import pandas as pd

df = pd.read_csv('C:\\Users\\Admin\\Desktop\\moodify\\AmazonProductReviewsDataset.csv\\7817_1.csv')

category_column = 'categories' 

unique_categories = df[category_column].value_counts()
df[category_column] = df[category_column].str.strip().str.lower()

print("Unique Product Categories and Counts:")
print(unique_categories)

total_unique_categories = df[category_column].nunique()
print(f"\nTotal number of unique product categories: {total_unique_categories}")
