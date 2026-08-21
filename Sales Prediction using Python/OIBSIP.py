# Sales Prediction using Python
# Oasis Infobyte Internship - Task 5

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv("Advertising.csv")

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Dataset information
print("\nDataset Information:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove unnecessary unnamed column if present
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# Visualize relationship between TV advertising and Sales
plt.figure(figsize=(8,5))

sns.scatterplot(
    x='TV',
    y='Sales',
    data=data
)

plt.title("TV Advertising vs Sales")

plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(6,4))

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# Split features and target
X = data.drop('Sales', axis=1)
y = data['Sales']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Model evaluation
print("\nR2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Take user input
print("\nEnter Advertising Budget Details:")

tv = float(input("Enter TV Advertising Budget: "))
radio = float(input("Enter Radio Advertising Budget: "))
newspaper = float(input("Enter Newspaper Advertising Budget: "))

# User input dataframe
user_data = pd.DataFrame(
    [[tv, radio, newspaper]],
    columns=X.columns
)

# Predict sales
predicted_sales = model.predict(user_data)

print("\nPredicted Sales:",
      round(predicted_sales[0], 2))