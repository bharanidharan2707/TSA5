import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

print("Downloading dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print(f"Dataset loaded with {len(df)} records.")
print("Training model...")

# Create a pipeline that first vectorizes the text and then applies Naive Bayes
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Train the model on the full dataset
model.fit(df['message'], df['label'])

# Save the model
model_path = 'spam_model.pkl'
joblib.dump(model, model_path)
print(f"Model trained and saved to {model_path} successfully.")
