"""
Flask Web Application for CNN Image Classification
Provides a web interface to upload images and get classification results.
"""

import os
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms
from model import CNN, get_model

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'best_model.pth'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model at module level (required for gunicorn/Render)
model = None
load_model_called = False


def load_model():
    """Load the trained CNN model."""
    global model
    if os.path.exists(MODEL_PATH):
        model = get_model(MODEL_PATH, device)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠ Warning: No trained model found at {MODEL_PATH}")
        print("  Please run 'python train.py' first to train the model.")
        model = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Preprocess image for CNN model input.
    Resizes to 32x32 and normalizes with CIFAR-10 statistics.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)


def classify_image(image_path):
    """
    Classify an image using the trained CNN model.
    Returns class name and confidence scores for all classes.
    """
    if model is None:
        return None, None, "Model not loaded. Please train the model first."
    
    try:
        # Preprocess and predict
        image_tensor = preprocess_image(image_path)
        probabilities = model.predict(image_tensor)
        
        # Get top prediction
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_class = CNN.CLASSES[predicted_idx.item()]
        confidence_value = confidence.item() * 100
        
        # Get all class probabilities
        all_probs = {
            CNN.CLASSES[i]: round(probabilities[0][i].item() * 100, 2)
            for i in range(len(CNN.CLASSES))
        }
        # Sort by probability
        all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return predicted_class, confidence_value, all_probs
        
    except Exception as e:
        return None, None, str(e)


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', classes=CNN.CLASSES)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400
    
    # Save and classify
    try:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        predicted_class, confidence, all_probs = classify_image(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        if predicted_class is None:
            return jsonify({'error': all_probs}), 500
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probs
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


# Load model at import time so gunicorn/Render can use it
load_model()
print("\n" + "="*50)
print("CNN Image Classifier - Flask App")
print("="*50)
print(f"Device: {device}")
print(f"Classes: {', '.join(CNN.CLASSES)}")
print("="*50)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\nStarting server at http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop\n")
    app.run(debug=False, host='0.0.0.0', port=port)
