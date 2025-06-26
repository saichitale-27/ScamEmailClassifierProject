import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os

# 1. Read CSV with correct encoding
data = pd.read_csv('spam/spam.csv', encoding='latin1')

# 2. Check actual columns and rename
data = data[['v1', 'v2']]
data.columns = ['label', 'message']  # Rename to standard names

# 3. Convert label to binary
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 4. Split features and labels
X = data['message']
y = data['label']

# 5. Train model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(X, y)

# 6. Save model
os.makedirs("spam", exist_ok=True)
joblib.dump(model, 'spam/spam_model.pkl')

print("✅ Model trained and saved as spam/spam_model.pkl")
