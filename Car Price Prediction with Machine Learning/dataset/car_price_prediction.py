import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv("car data.csv")

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Dataset information
print("\nDataset Information:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Convert categorical data into numerical form
data.replace({
    'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
    'Selling_type': {'Dealer': 0, 'Individual': 1},
    'Transmission': {'Manual': 0, 'Automatic': 1}
}, inplace=True)

# Create Car Age column
current_year = 2025
data['Car_Age'] = current_year - data['Year']

# Remove unnecessary columns
data = data.drop(['Car_Name', 'Year'], axis=1)

# Display updated dataset
print("\nUpdated Dataset:")
print(data.head())

# Split features and target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

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
print("\nEnter Car Details:")

present_price = float(input("Enter Present Price: "))
driven_kms = int(input("Enter Kilometers Driven: "))
owner = int(input("Enter Number of Previous Owners: "))
car_age = int(input("Enter Car Age: "))

fuel_type = int(input("Fuel Type (Petrol=0, Diesel=1, CNG=2): "))
selling_type = int(input("Selling Type (Dealer=0, Individual=1): "))
transmission = int(input("Transmission (Manual=0, Automatic=1): "))

# User input data
user_data = pd.DataFrame(
    [[present_price, driven_kms, fuel_type,
      selling_type, transmission, owner, car_age]],
    columns=X.columns
)

# Predict car price
predicted_price = model.predict(user_data)

print("\nPredicted Car Selling Price:",
      round(predicted_price[0], 2), "Lakhs")