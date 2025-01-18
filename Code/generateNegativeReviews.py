import random
import csv
import random

# Templates for negative reviews with placeholders
negative_review_templates = [
    "The {product} is {adjective}, completely {adjective2}.",
    "Worst {experience} ever, not worth the {money}.",
    "It {broke} after {time}, really {adjective}.",
    "Terrible {aspect}, do not {recommend}.",
    "Customer service was {adjective} and {adjective2}.",
    "The description is {adjective}, it's nothing like {advertised}.",
    "Completely {adjective} with this purchase.",
    "Shipping was {adjective}, and the {product} arrived {damaged}.",
    "A total {waste} of money, {avoid} at all costs.",
    "It doesn't {work} as expected, very {adjective}.",
    "The {item} was {defective}, had to return it {immediately}.",
    "Horrible {experience}, I'll never buy this {brand} again.",
    "The {material} feels {adjective} and {flimsy}.",
    "Performance is {adjective}, absolutely {adjective2}.",
    "It stopped {working} after {time}, not {reliable}.",
    "Overpriced and {underdelivered}, very {adjective}.",
    "The size is completely {wrong}, not true to {description}.",
    "Packaging was {adjective}, the {item} came {scratched}.",
    "Does not {meet expectations}, very {adjective}.",
    "Not worth the {hype}, extremely {adjective}."
]

# Templates for negative headings
negative_heading_templates = [
    "{adjective} {product}",
    "Completely {adjective2}",
    "Terrible {aspect}",
    "Worst {experience} Ever",
    "Not Worth the {money}",
    "Extremely {adjective}",
    "Poor {aspect}",
    "Defective {item}",
    "{adjective2} {product}",
    "{adjective} and {adjective2}",
    "Never Buy This {brand}",
    "{adjective} Quality",
    "Broken After {time}",
    "Highly {adjective}",
    "A Total {waste}",
    "{adjective} Packaging",
    "{adjective} Experience",
    "Low {aspect} Standards",
    "Avoid This {product}",
    "Overpriced and {adjective}"
]

# Possible values for placeholders
placeholders = {
    "product": ["item", "product", "purchase", "device", "order"],
    "adjective": ["terrible", "awful", "disappointing", "horrible", "bad"],
    "adjective2": ["useless", "frustrating", "low quality", "subpar", "misleading"],
    "experience": ["experience", "purchase", "decision", "choice"],
    "money": ["money", "effort", "time"],
    "broke": ["broke", "failed", "stopped working"],
    "time": ["one use", "a week", "a day", "a month"],
    "aspect": ["quality", "experience", "design", "service"],
    "recommend": ["recommend", "suggest", "trust"],
    "damaged": ["damaged", "broken", "scratched"],
    "waste": ["waste", "loss", "mistake"],
    "avoid": ["avoid", "stay away", "do not buy"],
    "work": ["work", "function", "perform"],
    "item": ["item", "product", "device", "order"],
    "defective": ["defective", "broken", "non-functional"],
    "immediately": ["immediately", "right away", "within days"],
    "brand": ["brand", "company", "seller"],
    "material": ["material", "build", "construction"],
    "working": ["working", "functioning", "performing"],
    "reliable": ["reliable", "durable", "trustworthy"],
    "underdelivered": ["underdelivered", "underperformed", "fell short"],
    "wrong": ["wrong", "inaccurate", "off"],
    "description": ["description", "advertisement", "listing"],
    "scratched": ["scratched", "damaged", "marred"],
    "meet expectations": ["meet expectations", "live up to the hype", "satisfy requirements"],
    "hype": ["hype", "expectations", "promise"],
    "advertised": ["advertised", "described", "promised"],
    "flimsy": ["flimsy", "weak", "delicate"],
    "working": ["working", "functioning", "performing"]
}

unique_reviews = set()
while len(unique_reviews) < 1000:
    template = random.choice(negative_review_templates)
    review = template.format(**{key: random.choice(values) for key, values in placeholders.items()})
    unique_reviews.add(review)

unique_headings = set()
while len(unique_headings) < 1000:
    template = random.choice(negative_heading_templates)
    heading = template.format(**{key: random.choice(values) for key, values in placeholders.items()})
    unique_headings.add(heading)

data_to_append = []
for review, heading in zip(unique_reviews, unique_headings):
    data_to_append.append([random.randint(1,3), heading, review, 0])

# Define CSV file path (ensure this file exists and has the correct columns)
csv_file_path = 'sorted_data.csv'

# Append data to the CSV file
with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # If the CSV file is empty or doesn't have headers, write the header first (adjust column names)
    if csvfile.tell() == 0:  # File is empty
        csv_writer.writerow(['star_rating', 'review_headline', 'review_body', 'Sentiment'])  # Adjust column names
    
    # Write the new rows of data
    csv_writer.writerows(data_to_append)
print("Data appended to sorted_data.csv")