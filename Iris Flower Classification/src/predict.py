# This is what you would use in a real application to classify a single flower.
import joblib
import pandas as pd

def predict_species(model_path, data):
    model = joblib.load(model_path)
    
    # Define the feature names to match what the model saw during training
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    # Convert the list into a DataFrame with column names
    data_df = pd.DataFrame([data], columns=feature_names)
    
    return model.predict(data_df)[0]