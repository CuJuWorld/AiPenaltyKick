import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

class BallKickingPowerModel:
    def __init__(self, features_path, labels_path):
        self.features_path = features_path
        self.labels_path = labels_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def load_data(self):
        features = pd.read_csv(self.features_path)
        labels = pd.read_csv(self.labels_path)
        return features, labels

    def train(self):
        features, labels = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        print(f'Model trained with Mean Squared Error: {mse}')

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)
        print(f'Model saved to {model_path}')

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
        print(f'Model loaded from {model_path}')

    def predict(self, new_data):
        return self.model.predict(new_data)