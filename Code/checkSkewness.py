import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import chardet

with open('sorted_data.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
print(f"Detected file encoding: {encoding}")

# Load the data
df = pd.read_csv('sorted_data.csv', encoding=encoding)

# Select a column for analysis (replace 'column_name' with your actual column)
column_name = "Sentiment"
data = df[column_name].dropna()  # Drop NaN values

# Calculate skewness
skewness_value = skew(data)
print(f"Skewness of {column_name}: {skewness_value:.2f}")

# Plot histogram with KDE (Kernel Density Estimate)
plt.figure(figsize=(8, 6))
sns.histplot(data, bins=30, kde=True, color="blue")
plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(data.median(), color='green', linestyle='dashed', linewidth=2, label='Median')

# Add legend and title
plt.legend()
plt.title(f"Distribution of {column_name} (Skewness: {skewness_value:.2f})")
plt.xlabel(column_name)
plt.ylabel("Frequency")

plt.show()