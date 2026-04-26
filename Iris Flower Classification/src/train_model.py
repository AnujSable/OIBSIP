# This script trains the Random Forest and saves it to the models/ folder.
 

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_save(X_train, y_train, model_path):
    # Initialize and train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model