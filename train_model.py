import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Add labels
df_fake["label"] = 0
df_real["label"] = 1

# Combine
df = pd.concat([df_fake, df_real])
df = df.sample(frac=1, random_state=42)

# Use only text column
X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")
