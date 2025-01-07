import csv

# Define the input and output file paths
input_file = 'C:\\Moodify\\cleaned_data.csv'
output_file = 'C:\\Moodify\\sorted_data.csv'

# Specify the columns to copy (0-indexed positions)
columns_to_copy = [5, 6, 7, 12, 13]  # Example: copying the 1st and 3rd columns
rating_column_index = 7   # Example: 4th column in the original CSV (0-indexed)

with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Add a header row if the input CSV has headers
    header = next(reader)
    new_header = [header[i] for i in columns_to_copy] + ["Sentiment"]
    writer.writerow(new_header)

    for row in reader:
        # Extract selected columns
        selected_columns = [row[i] for i in columns_to_copy]

        # Determine the value of the new column based on the rating
        if row[rating_column_index] == '':
            continue
        rating = float(row[rating_column_index])
        if rating < 3:
            rating_category = -1
        elif rating == 3:
            rating_category = 0
        else:
            rating_category = 1

        # Append the new column to the row
        selected_columns.append(rating_category)

        # Write the modified row to the new CSV
        writer.writerow(selected_columns)

print(f"Selected columns with rating category added to {output_file}")
#dataset link
#https://www.kaggle.com/datasets/miriamodeyianypeter/sentiment-analysis-amazon-product-reviews