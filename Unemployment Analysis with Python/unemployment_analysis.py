import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("Unemployment in India.csv")

# Display first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# Dataset information
print("\nDataset Information:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Display column names
print("\nColumn Names:")
print(data.columns)

# Convert Date column into datetime format
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Unemployment Rate by Region
plt.figure(figsize=(14, 6))

sns.barplot(
    x='Region',
    y='Estimated Unemployment Rate (%)',
    data=data,
    palette='viridis'
)

plt.xticks(rotation=75)
plt.title("Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")

plt.tight_layout()
plt.show()

# Unemployment Trend Over Time
plt.figure(figsize=(12, 6))

sns.lineplot(
    x='Date',
    y='Estimated Unemployment Rate (%)',
    data=data
)

plt.title("Unemployment Rate Over Time")

plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(8, 5))

numeric_data = data.select_dtypes(include=['float64', 'int64'])

sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Average unemployment rate
average_rate = data['Estimated Unemployment Rate (%)'].mean()

print("\nAverage Unemployment Rate:", average_rate)

# Region with highest unemployment
highest_region = data.groupby('Region')[
    'Estimated Unemployment Rate (%)'
].mean().sort_values(ascending=False)

print("\nAverage Unemployment Rate by Region:")
print(highest_region)