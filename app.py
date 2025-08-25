import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json


# Initialize backend using Flask API
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from frontend

# Load dataset once at startup
df = pd.read_csv('house_data.csv')

# Load selected features from JSON once at startup
with open('features/selected_features.json') as f:
    selected_features_map = json.load(f)

# Load models used from JSON once at startup
with open('models/model_names.json') as f:
    valid_models = json.load(f)


# Load sample data for user to use
@app.route('/sample/<model>', methods=['GET'])
def get_sample_input(model):
    """
    Returns a random row from the dataset with only the selected features for the model.
    """
    model_key = model.replace("_", " ").title()
    selected_features = selected_features_map.get(model_key)

    if not selected_features:
        return jsonify({'error': f'Model "{model_key}" not found'}), 404

    # Pick a random row
    sample_row = df.sample(1).iloc[0]

    # Extract only the selected features
    # sample_input = {feature: sample_row[feature].item() for feature in selected_features if feature in sample_row}
    sample_input = {
        feature: sample_row[feature].item() if hasattr(sample_row[feature], 'item') else sample_row[feature]
        for feature in selected_features if feature in sample_row
    }

    return jsonify(sample_input)


# endpoint to retrieve all models
@app.route('/models', methods=['GET'])
def get_models():
    return jsonify(valid_models)


# endpoint to retrieve selected feature for a model
@app.route('/features/<model>', methods=['GET'])
def get_model_features(model):
    model_key = model.replace("_", " ").title()
    features = selected_features_map.get(model_key)
    if not features:
        return jsonify({'error': f'Model "{model_key}" not found'}), 404
    return jsonify(features)


# endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives JSON with model name and input features.
    Loads the corresponding model, scaler, and feature list.
    Returns predicted price and simulated similar prices.
    """

    data = request.json
    model_name = data['model']
    input_features = data['features']

    try:
        # Normalize model name to match keys in JSON
        model_key = model_name.replace("_", " ").title()
        if model_key not in valid_models:
            return jsonify({'error': f'Model "{model_key}" is not supported.'}), 400

        # Get selected features for the model
        selected_features = selected_features_map.get(model_key)
        if not selected_features:
            return jsonify({'error': f'Model "{model_key}" not found in selected_features.json'}), 400

        # Load model and scaler
        model = joblib.load(f'models/{model_name}.pkl')
        scaler = joblib.load(f'scalers/{model_name}_scaler.pkl')

        # Build input vector in correct order
        input_vector = np.array([input_features[feature] for feature in selected_features]).reshape(1, -1)

        # Scale and predict
        scaled_input = scaler.transform(input_vector)
        prediction = model.predict(scaled_input)[0]

        # Simulate similar prices for charting
        similar_prices = list(np.random.normal(loc=prediction, scale=50000, size=20))

        return jsonify({
            'predicted_price': float(prediction),
            'similar_prices': similar_prices
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
