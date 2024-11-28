from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
from rembg import remove
import io
import json
from datetime import datetime
import tensorflow as tf
import os
import requests
from ..utils.config import (
    INITIAL_MODEL_PATH, MODEL_DIR,
    API_CONFIG, MODEL_TEMPLATES,
    INTERFACE_DIR
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
                'last_modified': datetime.fromtimestamp(
                    os.path.getmtime(path)
                ).strftime('%Y-%m-%d %H:%M:%S')
            })
    def get_priority(model):
        name = model['name'].lower()
        if 'best_additional_model' in name:
            return 0  # Ưu tiên cao nhất
        elif 'best_initial_model' in name:
            return 1  # Ưu tiên thứ hai
        return 2     # Các file còn lại

    # Sắp xếp theo priority trước, sau đó mới đến last_modified
    return sorted(models, key=lambda x: (get_priority(x), 
                        -datetime.strptime(x['last_modified'], '%Y-%m-%d %H:%M:%S').timestamp()))

def load_model_by_name(model_name):
    """Load model theo tên."""
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"Model {model_name} does not exist")
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

def get_dataset_statistics():
    """Lấy thống kê về tập train và test của từng client."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Tính toán thống kê cho từng client
    client_stats = {}
    
    # Client 0: digits 0-2
    train_mask_0 = np.isin(y_train, [0, 1, 2])
    test_mask_0 = np.isin(y_test, [0, 1, 2])
    train_labels_0 = y_train[train_mask_0]
    test_labels_0 = y_test[test_mask_0]
    
    client_stats[0] = {
        'train_samples': len(train_labels_0),
        'test_samples': len(test_labels_0),
        'train_distribution': {str(i): int(np.sum(train_labels_0 == i)) for i in [0, 1, 2]},
        'test_distribution': {str(i): int(np.sum(test_labels_0 == i)) for i in [0, 1, 2]}
    }
    
    # Client 1: digits 3-4
    train_mask_1 = np.isin(y_train, [3, 4])
    test_mask_1 = np.isin(y_test, [3, 4])
    train_labels_1 = y_train[train_mask_1]
    test_labels_1 = y_test[test_mask_1]
    
    client_stats[1] = {
        'train_samples': len(train_labels_1),
        'test_samples': len(test_labels_1),
        'train_distribution': {str(i): int(np.sum(train_labels_1 == i)) for i in [3, 4]},
        'test_distribution': {str(i): int(np.sum(test_labels_1 == i)) for i in [3, 4]}
    }
    
    return client_stats

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
            # Lưu model mới và thống kê dataset
            model.save(INITIAL_MODEL_PATH)
            dataset_stats = get_dataset_statistics()
            stats_path = os.path.join(MODEL_DIR, 'dataset_statistics.json')
            with open(stats_path, 'w') as f:
                json.dump(dataset_stats, f, indent=4)
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
        all_confidence = []
        for i in prediction[0]:
            all_confidence.append(round(float(i),6))
        
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
            'confidence': round(confidence*100,4),
            'all_confidence': all_confidence,
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
        dataset_stats = get_dataset_statistics()
        return jsonify({
            'status': 'healthy',
            'available_models': models,
            'total_models': len(models),
            'dataset_statistics': dataset_stats,
            'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500
    
@app.route('/model-stats/<model_name>', methods=['GET'])
def get_stats(model_name):
    """API endpoint để lấy thống kê model."""
    json_name = model_name.replace('.keras', '.json')
    json_path = os.path.join(MODEL_DIR, "results", json_name)
    
    # Check if file exists using os.path.exists instead of .exists()
    if not os.path.exists(json_path):
        return jsonify({
            'error': 'Model statistics not found',
            'status': f'No statistics found for model: {model_name}'
        }), 404
        
    # Read and parse JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract required statistics
    training_info = data['training_info']
    dataset_stats = data['dataset_statistics']
    active_clients = data['active_clients_info']
    
    # Compile client labels
    client_labels = []
    for _, client_info in active_clients['client_ranges'].items():
        client_labels = client_labels + client_info['labels']
    client_labels = list(dict.fromkeys(client_labels))
    
    # Calculate accuracy per round
    accuracy_per_round = [
        {
            'round': round['round'],
            'accuracy': round['accuracy'],  # Convert to percentage
            'client_accuracies': {
                client['client_id']: client['accuracy']
                for client in round['client_metrics']
            }
        }
        for round in training_info['training_history']
    ]
    
    # Compile statistics
    return jsonify({
        'model_name': model_name,
        'total_rounds': training_info['total_rounds'],
        'final_metrics': {
            'accuracy': training_info['final_accuracy'],  # Convert to percentage
            'loss': training_info['final_loss']
        },
        'client_labels': client_labels,
        'dataset_size': {
            'train': dataset_stats['overall']['train']['total_samples'],
            'test': dataset_stats['overall']['test']['total_samples']
        },
        'accuracy_history': accuracy_per_round,
        'timestamp': data["timestamp"],
    })

@app.route('/')
def serve_vue_app():
    return send_from_directory(INTERFACE_DIR, 'index.html')

@app.route('/<path:path>')
def send_js(path):
    return send_from_directory(INTERFACE_DIR, path)

if __name__ == '__main__':
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )