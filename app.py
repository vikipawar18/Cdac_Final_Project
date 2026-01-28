import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from prediction import predict_video

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Use the rebuilt model that's compatible with TensorFlow 2.20.0
MODEL_PATH = 'deepfake_model_rebuilt.h5'
if not os.path.exists(MODEL_PATH):
    # Fallback to other models if they exist
    if os.path.exists('best_deepfake_model.h5'):
        MODEL_PATH = 'best_deepfake_model.h5'
    elif os.path.exists('mask_detector_model.h5'):
        MODEL_PATH = 'mask_detector_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Call the prediction function
            # Pass the model path explicitly
            label, confidence = predict_video(filepath, model_path=MODEL_PATH)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify({
                'label': label,
                'confidence': float(confidence)
            })
            
        except Exception as e:
            # Clean up file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
