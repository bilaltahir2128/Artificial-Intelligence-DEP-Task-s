import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')

# Load and shuffle movie reviews data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Join words into sentences and create labels
data = [" ".join(words) for words, category in documents]
labels = [category for words, category in documents]

# Split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer(binary=True)
train_data_vec = vectorizer.fit_transform(train_data)
test_data_vec = vectorizer.transform(test_data)

# Train a logistic regression model
model = LogisticRegression()
model.fit(train_data_vec, train_labels)

# Predict and evaluate
predictions = model.predict(test_data_vec)
accuracy = accuracy_score(test_labels, predictions)

print(f"Accuracy: {accuracy:.2f}")

# Function for predicting sentiment of new text
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

# Example usage
new_text = "I really enjoyed this movie. It was fantastic!"
print(f"Sentiment: {predict_sentiment(new_text)}")

