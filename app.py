from flask import Flask, request, jsonify
from flask_cors import CORS 
import joblib
import numpy as np

app = Flask(__name__)
CORS(app) # Allows frontend running on a different port/address to access this API

# --- Load Resources ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl') # Use saved names for correct order
    print("Resources loaded successfully.")
except Exception as e:
    print(f"Error loading resources. Did you run train_model.py? Error: {e}")
    model = None
    scaler = None
    feature_names = []
# ----------------------

def get_risk_level(probability):
    """Simple risk categorization based on prediction probability."""
    if probability >= 0.75:
        return "High Risk"
    elif probability >= 0.5:
        return "Moderate Risk"
    else:
        return "Low Risk"

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the dashboard."""
    if model is None or scaler is None:
        return jsonify({'success': False, 'error': 'Model not loaded.'}), 500

    data = request.get_json(force=True)
    
    try:
        # Extract features in the correct order using the saved feature_names list
        features = [data[key] for key in feature_names]
        
        # Reshape and Scale
        input_array = np.array(features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        
        # Get Probability (0 = No Diabetes, 1 = Diabetes)
        probability_class_1 = model.predict_proba(scaled_input)[0][1] 
        
        # Determine the final risk message
        risk_level = get_risk_level(probability_class_1)
        
        # Return the results to the frontend
        return jsonify({
            'success': True,
            'risk_level': risk_level,
            'probability': round(probability_class_1 * 100, 1), 
            'message': 'Assessment Complete'
        })

    except KeyError:
        return jsonify({'success': False, 'error': 'Missing data fields in request.'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'An unknown error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    print("Flask API running. Access at http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)