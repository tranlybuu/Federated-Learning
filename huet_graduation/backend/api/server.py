from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
from ..utils.config import INITIAL_MODEL_PATH, GLOBAL_MODEL_TEMPLATE, MODEL_DIR, API_CONFIG
from ..federated_learning.model import create_model

app = Flask(__name__)
CORS(app)

def get_latest_model_path():
    """Lấy model mới nhất từ các rounds training."""
    try:
        # Tìm tất cả các file model trong thư mục
        model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('global_model_round_')]
        if not model_files:
            return INITIAL_MODEL_PATH
        
        # Lấy round number từ tên file
        round_numbers = [int(f.split('_')[-1].replace('.keras', '')) for f in model_files]
        latest_round = max(round_numbers)
        
        return GLOBAL_MODEL_TEMPLATE.format(latest_round)
    except Exception as e:
        print(f"Error finding latest model: {e}")
        return INITIAL_MODEL_PATH

def load_or_create_model():
    """Load model đã train hoặc tạo model mới nếu chưa có."""
    try:
        model_path = get_latest_model_path()
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            return tf.keras.models.load_model(model_path)
        else:
            print(f"Creating new model as {model_path} does not exist")
            model = create_model()
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            # Lưu model mới
            model.save(INITIAL_MODEL_PATH, save_format='keras')
            return model
    except Exception as e:
        print(f"Error loading/creating model: {e}")
        # Trong trường hợp lỗi, tạo model mới
        model = create_model()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

def preprocess_image(image_data):
    """Xử lý ảnh trước khi đưa vào model."""
    try:
        # Chuyển đổi bytes thành ảnh
        image = Image.open(io.BytesIO(image_data))
        
        # Chuyển sang grayscale nếu cần
        if image.mode != 'L':
            image = image.convert('L')
            
        # Resize về kích thước 28x28
        if image.size != (28, 28):
            image = image.resize((28, 28))
            
        # Chuyển thành numpy array và normalize
        image_array = np.array(image).astype('float32')
        image_array = image_array.reshape(1, 28, 28, 1) / 255.0
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

# Load model khi khởi động server
model = load_or_create_model()

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        # Nhận và xử lý ảnh
        image_data = request.data
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
            
        image_array = preprocess_image(image_data)
        
        # Dự đoán
        prediction = model.predict(image_array)
        digit = np.argmax(prediction[0])
        confidence = float(prediction[0][digit])
        
        return jsonify({
            'digit': int(digit),
            'confidence': confidence,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in recognition: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint kiểm tra trạng thái server."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': get_latest_model_path()
    })

if __name__ == '__main__':
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )