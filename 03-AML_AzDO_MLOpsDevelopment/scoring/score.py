import os
import numpy as np
import pandas as pd
import tensorflow.keras
import joblib
from tensorflow.keras.models import load_model
import h5py

def init():
    global model
    global scaler
    global init_error
    
    try:

        init_error = None

        scaler_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model_files', 'scaler.pkl')
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model_files', 'anomaly_detection_encoder_model.h5')
        
        print('Loading scaler from:', scaler_path)
        scaler = joblib.load(scaler_path)

        print('Loading model from:', model_path)
        model = load_model(model_path)

    except Exception as e:
        init_error = e
        print(e)
        
# note you can pass in multiple rows for scoring
def run(raw_data):

    if init_error is not None:
        return 'Init error: {}'.format(str(init_error))

    try:
        print("Received input:", raw_data)
    
        input_df = pd.read_json(raw_data['data'], orient='values')
    
        sensor_readings = np.array(input_df)
        scaled_sensor_readings = scaler.transform(sensor_readings.reshape(1,-1))

        pred_sensor_readings = model.predict(scaled_sensor_readings)
        score = np.mean(np.abs(scaled_sensor_readings - pred_sensor_readings[0]))

    if score > 0.01:
        print('WARNING! Abnormal conditions detected')
        return 1
    else:
        print('Everything is ok')
        return 0

    except Exception as e:
        error = str(e)
        return error


if __name__ == "__main__":
    # Test scoring
    init()
    test_row = '{"data":[[70, 200, 60.6, 0, 1448.17],[14.23, 41, 14.4, 318.50, 601.95]]}'
    prediction = run(test_row, {})
    print("Test result: ", prediction)
