import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Remove Id and separate features/target
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    # Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test