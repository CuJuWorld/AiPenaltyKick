import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(features_path, labels_path, model_save_path):
    # Load the features and labels
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_save_path)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f'Model R^2 score: {score}')

if __name__ == "__main__":
    train_model('../data/processed/features.csv', '../data/labels/kicking_power_labels.csv', '../models/kicking_power_model.pkl')