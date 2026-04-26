import pandas as pd
from src.data_preprocessing import prepare_data
from src.train_model import train_and_save
from src.predict import predict_species
from src.visualize import save_plots # New import for plots

# 1. Process Data
# This handles the loading and splitting logic from your preprocessing script
X_train, X_test, y_train, y_test = prepare_data('data/Iris.csv')

# 2. Train and Save
# This trains the Random Forest and saves the .pkl file to the models/ folder
model_file = 'models/iris_rf_model.pkl'
model = train_and_save(X_train, y_train, model_file)

# 3. Visualize Results (New Step)
# This creates the charts in your outputs/ folder
y_pred = model.predict(X_test)
# We load the full dataframe once for the pairplot visualization
full_df = pd.read_csv('data/Iris.csv').drop('Id', axis=1, errors='ignore')
save_plots(full_df, y_test, y_pred, model.classes_)

# 4. Predict
# Testing a single flower sample
test_flower = [5.1, 3.5, 1.4, 0.2]
result = predict_species(model_file, test_flower)

print("-" * 30)
print(f"The predicted species for {test_flower} is: {result}")
print("-" * 30)
print("Project execution complete. Check the 'outputs/' folder for your plots.")