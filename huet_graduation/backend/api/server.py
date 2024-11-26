from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from rembg import remove
import io
from datetime import datetime
import tensorflow as tf
import os
import requests
from ..utils.config import (
    INITIAL_MODEL_PATH, MODEL_DIR,
    API_CONFIG, MODEL_TEMPLATES
)
from ..federated_learning.model import create_model

app = Flask(__name__)
CORS(app)

def get_available_models():
    """Lấy danh sách tất cả các model có sẵn."""
    models = []
    for file in os.listdir(MODEL_DIR):
        if file.endswith('.keras'):
            path = os.path.join(MODEL_DIR, file)
            models.append({
                'name': file,
                'path': path,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
            })
    return sorted(models, key=lambda x: x['last_modified'], reverse=True)

def load_model_by_name(model_name):
    """Load model theo tên."""
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"Model {model_name} không tồn tại")
    return tf.keras.models.load_model(model_path)

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
        
        return MODEL_TEMPLATES['global'].format(latest_round)
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
            model.save(INITIAL_MODEL_PATH)
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
        
        # Chuyển sang RGBA nếu cần
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Xoá nền bằng thư viện rembg
        image_no_bg = remove(image)  # Xoá nền
        
        # Chuyển đổi ảnh đã xoá nền sang grayscale
        image_no_bg = image_no_bg.convert('L')
        
        # Resize về kích thước 28x28
        if image_no_bg.size != (28, 28):
            image_no_bg = image_no_bg.resize((28, 28))
        
        # Chuyển thành numpy array và normalize
        image_array = np.array(image_no_bg).astype('float32').reshape(1, 28, 28, 1) / 255.0
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

# Load model khi khởi động server
model = load_or_create_model()

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        # Kiểm tra và lấy model được chỉ định
        model_name = request.args.get('model')
        if model_name:
            try:
                model = load_model_by_name(model_name)
                print(f"Using specified model: {model_name}")
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
        else:
            model_name = os.path.basename(get_latest_model_path())
            print(f"Using latest model: {model_name}")

        # Kiểm tra xem có dữ liệu URL không
        if request.is_json:
            data = request.get_json()
            image_url = data.get('url')
            if image_url:
                image_data = requests.get(image_url).content
            else:
                return jsonify({'error': 'No image URL provided'}), 400
        else:
            image_data = request.data
            if not image_data:
                return jsonify({'error': 'No image data received'}), 400
            
        # Xử lý ảnh và dự đoán
        image_array = preprocess_image(image_data)
        prediction = model.predict(image_array)
        digit = np.argmax(prediction[0])
        confidence = float(prediction[0][digit])
        
        # Chuẩn bị thông tin response
        model_info = {
            'name': model_name,
            'path': os.path.join(MODEL_DIR, model_name),
            'last_modified': datetime.fromtimestamp(
                os.path.getmtime(os.path.join(MODEL_DIR, model_name))
            ).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify({
            'digit': int(digit),
            'confidence': confidence,
            'success': True,
            'model_info': model_info,
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
    try:
        models = get_available_models()
        return jsonify({
            'status': 'healthy',
            'available_models': models,
            'total_models': len(models),
            'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )