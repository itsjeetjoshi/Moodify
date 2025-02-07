import random
import csv
import random

neutral_review_templates = [
    "The {product} is {adjective}, nothing too {adjective2}.",
    "An {average} {experience}, neither {good} nor {bad}.",
    "It {works} as expected, but {nothing_special}.",
    "Decent {aspect}, but could be {better}.",
    "Customer service was {adjective}, took {time} to resolve.",
    "The description is {mostly_accurate}, but some details were {off}.",
    "An {okay} purchase, but {some_issues}.",
    "Shipping was {adjective}, and the {product} arrived {fine}.",
    "A {fair} product for the {price}, but {not_excellent}.",
    "It does {what_it_says}, but {lacks_features}.",
    "The {item} was {functional}, but had {minor_issues}.",
    "A {decent} {experience}, nothing {remarkable}.",
    "The {material} feels {adjective}, but {sturdy}.",
    "Performance is {adjective}, but not {outstanding}.",
    "It {lasted} {longer_than_expected}, but {not_perfect}.",
    "Reasonably priced, but {could_be_better}.",
    "The size is {close} to {description}, but {slightly_off}.",
    "Packaging was {adjective}, the {item} came {intact}.",
    "Meets {basic_expectations}, but {could_improve}.",
    "Worth considering if you need a {functional} {product}."
]

neutral_heading_templates = [
    "{adjective} {product}",
    "An {average} {experience}",
    "{adjective2}, But Not {amazing}",
    "{decent} Quality",
    "{reasonable} for the {price}",
    "Not {bad}, Not {great}",
    "{functional} {item}",
    "{mostly_accurate} Description",
    "{fair} Deal",
    "{adjective} Performance",
    "{adjective} Packaging",
    "A {neutral} {experience}",
    "Satisfactory {aspect}",
    "{decent}, But {nothing_special}",
    "Works, But Could Be {better}",
    "{reliable}, But Not {outstanding}",
    "Close to {expectations}",
    "{fine}, But {lacks_features}",
    "Mediocre {product}",
    "Average {performance}"
]

placeholders = {
    "product": ["item", "product", "purchase", "device", "order"],
    "adjective": ["decent", "okay", "fair", "satisfactory", "reasonable"],
    "adjective2": ["acceptable", "average", "mediocre", "fine", "not bad"],
    "experience": ["experience", "purchase", "decision", "choice"],
    "good": ["good", "great", "amazing"],
    "bad": ["bad", "disappointing", "underwhelming"],
    "works": ["works", "functions", "operates"],
    "nothing_special": ["nothing special", "nothing remarkable", "nothing extraordinary"],
    "aspect": ["quality", "experience", "design", "service"],
    "better": ["better", "improved", "enhanced"],
    "time": ["some time", "a few days", "a week"],
    "mostly_accurate": ["mostly accurate", "fairly accurate", "somewhat true"],
    "off": ["off", "inaccurate", "slightly misleading"],
    "okay": ["okay", "fine", "passable"],
    "some_issues": ["some issues", "minor flaws", "slight concerns"],
    "fine": ["fine", "undamaged", "acceptable"],
    "fair": ["fair", "reasonable", "justifiable"],
    "price": ["price", "cost", "value"],
    "not_excellent": ["not excellent", "not outstanding", "not impressive"],
    "what_it_says": ["what it says", "what’s advertised", "the job"],
    "lacks_features": ["lacks features", "is basic", "could use more options"],
    "item": ["item", "product", "device", "order"],
    "functional": ["functional", "usable", "working"],
    "minor_issues": ["minor issues", "small defects", "cosmetic problems"],
    "remarkable": ["remarkable", "exceptional", "memorable"],
    "material": ["material", "build", "construction"],
    "sturdy": ["sturdy", "durable", "solid"],
    "outstanding": ["outstanding", "exceptional", "remarkable"],
    "lasted": ["lasted", "held up", "remained functional"],
    "longer_than_expected": ["longer than expected", "as expected", "decently long"],
    "not_perfect": ["not perfect", "has flaws", "could improve"],
    "could_be_better": ["could be better", "could improve", "needs enhancement"],
    "close": ["close", "similar", "almost identical"],
    "description": ["description", "advertisement", "listing"],
    "slightly_off": ["slightly off", "a bit inaccurate", "not precise"],
    "intact": ["intact", "undamaged", "in good condition"],
    "basic_expectations": ["basic expectations", "minimum requirements", "the essentials"],
    "could_improve": ["could improve", "needs work", "could be refined"],
    "functional": ["functional", "practical", "useful"],
    "neutral": ["neutral", "balanced", "fair"],
    "reliable": ["reliable", "consistent", "dependable"],
    "expectations": ["expectations", "standards", "advertised claims"],
    "fine": ["fine", "acceptable", "decent"],
    "decent": ["decent", "average", "not bad"],
    "average": ["decent", "average",],
    "great": ["good", "great"],
    "amazing": ["amazing", "great"],
    "performance": ["performance"],
    "reasonable": ["reasonable"]
}

unique_reviews = set()
while len(unique_reviews) < 100:
    template = random.choice(neutral_review_templates)
    review = template.format(**{key: random.choice(values) for key, values in placeholders.items()})
    unique_reviews.add(review)

unique_headings = set()
while len(unique_headings) < 100:
    template = random.choice(neutral_heading_templates)
    heading = template.format(**{key: random.choice(values) for key, values in placeholders.items()})
    unique_headings.add(heading)

data_to_append = []
for review, heading in zip(unique_reviews, unique_headings):
    data_to_append.append(['Fire HD 7, 7" HD Display, Wi-Fi, 8 GB', 'PC', 3, heading, review, 0])

csv_file_path = 'sorted_data.csv'

with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    if csvfile.tell() == 0:
        csv_writer.writerow(['product_title', 'product_category', 'star_rating', 'review_headline', 'review_body', 'Sentiment'])
    csv_writer.writerows(data_to_append)
print("Data appended to sorted_data.csv")