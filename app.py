from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('stacking_classifier.joblib')

@app.route('/')
def home():
    return "Welcome to the Model Prediction App!"

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the POST request
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    
    # Predict using the model
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]  # Assuming binary classification

    # Return the prediction
    return jsonify({'prediction': int(prediction[0]), 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)