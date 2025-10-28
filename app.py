from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ======================================================
# ⚙️ Flask App Setup
# ======================================================
app = Flask(__name__)

# ======================================================
# 🔧 Model & Scaler Paths (Auto-detect)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try both possible locations
MODEL_PATH_1 = os.path.join(BASE_DIR, 'lstm_energy_forecast.h5')
MODEL_PATH_2 = os.path.join(BASE_DIR, 'model', 'lstm_energy_forecast_model.h5')

SCALER_PATH_1 = os.path.join(BASE_DIR, 'scaler.pkl')
SCALER_PATH_2 = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

# Select whichever exists
MODEL_PATH = MODEL_PATH_1 if os.path.exists(MODEL_PATH_1) else MODEL_PATH_2
SCALER_PATH = SCALER_PATH_1 if os.path.exists(SCALER_PATH_1) else SCALER_PATH_2

# ======================================================
# 📦 Load Model & Scaler
# ======================================================
try:
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Model & Scaler loaded successfully from:\n   {MODEL_PATH}\n   {SCALER_PATH}")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    model, scaler = None, None

# ======================================================
# 🏠 Home Route
# ======================================================
@app.route('/')
def home():
    return render_template('index.html')

# ======================================================
# 🔮 Forecast Route
# ======================================================
@app.route('/forecast', methods=['POST'])
def forecast():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded properly. Please check paths.'})

    try:
        data = request.get_json()
        values = np.array(data['data']).reshape(-1, 1)
        horizon = int(data.get('horizon', 24))

        # Scale input
        scaled = scaler.transform(values)

        # Define look_back (same as during training)
        look_back = 24
        if len(scaled) < look_back:
            return jsonify({'error': f'Not enough data points. Need at least {look_back} readings.'})

        x_input = scaled[-look_back:].reshape(1, look_back, 1)

        preds = []
        last_input = x_input.copy()

        for _ in range(horizon):
            next_pred = model.predict(last_input, verbose=0)[0][0]
            preds.append(next_pred)
            last_input = np.append(last_input[:, 1:, :], [[[next_pred]]], axis=1)

        # Inverse transform predictions
        preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

        # Generate timestamps for the next hours
        timestamps = [(pd.Timestamp.now() + pd.Timedelta(hours=i + 1)).isoformat() for i in range(horizon)]
        forecast_data = [{'t': t, 'y': float(v)} for t, v in zip(timestamps, preds_inv)]

        return jsonify({'forecast': forecast_data})

    except Exception as e:
        print("❌ Forecast error:", e)
        return jsonify({'error': str(e)})

# ======================================================
# 🚀 Run Flask App
# ======================================================
if __name__ == '__main__':
    app.run(debug=True)






