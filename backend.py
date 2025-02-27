from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

import gdown
MODEL_PATH = "classification_model.keras"
DRIVE_FILE_ID = "1d_W3A55rVuK467iNwmpPVPmuZ7YcLa2t"

app = Flask(__name__)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model . . .")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url,MODEL_PATH,quiet=False)
    else:
        print("Model already exists, skipping download.")

download_model()
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.resize((256,256))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image
 
@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(
            {'error':'No file uploaded'}
        ),400

    file = request.files['file']
    if file.filename == '':
        return jsonify(
            {'error':'No selected file'}
        ),400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        label = "Dog" if prediction[0][0] > 0.5 else "Cat"
        
        return jsonify(
            {'prediction':label}
            )

    except Exception as e:
        return jsonify({
            'error':str(e)
        }),500

if __name__ == '__main__':
    port = int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)

