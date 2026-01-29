import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from prediction import predict_video
from functools import wraps

app = Flask(__name__)
app.secret_key = os.urandom(24) # Secret key for sessions

# Mock user database
users = {
    "admin": "password123"
}

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

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if username in users and users[username] == password:
            session['user'] = username
            return jsonify({'message': 'Login successful'})
        else:
            return jsonify({'error': 'Invalid username or password'}), 401
            
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    name = data.get('name')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
        
    if username in users:
        return jsonify({'error': 'Username already exists'}), 400
        
    users[username] = password
    session['user'] = username
    return jsonify({'message': 'Signup successful'})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

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
