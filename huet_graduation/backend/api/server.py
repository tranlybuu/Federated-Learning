from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
from federated_learning.model import create_model
from utils.config import MODEL_PATH, API_CONFIG
import os
import tensorflow as tf

app = Flask(__name__)
CORS(app) 

def load_or_create_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print(f"Creating new model as {MODEL_PATH} does not exist")
        model = create_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

# Load the trained model
model = load_or_create_model()

@app.route('/recognize', methods=['POST'])
def recognize():
    image_data = request.data
    image = Image.open(io.BytesIO(image_data))
    
    # Preprocess the image
    image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
    
    # Make prediction
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)
    confidence = float(prediction[0][digit])
    
    return jsonify({'digit': int(digit), 'confidence': confidence})

if __name__ == '__main__':
    app.run(host=API_CONFIG['host'], port=API_CONFIG['port'], debug=API_CONFIG['debug'])