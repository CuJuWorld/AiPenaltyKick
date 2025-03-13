import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def evaluate_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    evaluation_results = {
        'Mean Squared Error': mse,
        'R-squared': r2
    }
    
    return evaluation_results

def main():
    # Load test data
    X_test = pd.read_csv('../data/processed/features.csv')
    y_test = pd.read_csv('../data/labels/kicking_power_labels.csv')

    # Evaluate ball speed model
    ball_speed_results = evaluate_model('../models/ball_speed_model.pkl', X_test, y_test['ball_speed'])
    print("Ball Speed Model Evaluation:")
    print(ball_speed_results)

    # Evaluate kicking power model
    kicking_power_results = evaluate_model('../models/kicking_power_model.pkl', X_test, y_test['kicking_power'])
    print("Kicking Power Model Evaluation:")
    print(kicking_power_results)

if __name__ == "__main__":
    main()