import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Keep only necessary columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['Category', 'Message']

# Display updated dataset
print("\nUpdated Dataset:")
print(data.head())

# Convert category into numerical values
data['Category'] = data['Category'].map({
    'ham': 0,
    'spam': 1
})

# Split features and target
X = data['Message']
y = data['Category']

# Convert text into numerical vectors
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Take user input
print("\nEnter Email Message:")

user_message = input()

# Convert input into vector
user_vector = vectorizer.transform([user_message])

# Predict spam or ham
prediction = model.predict(user_vector)

# Display result
if prediction[0] == 1:
    print("\nThis message is SPAM.")
else:
    print("\nThis message is NOT SPAM.")