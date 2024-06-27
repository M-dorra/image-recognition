from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__, template_folder='.')

# Load the model
model = ResNet50(weights='imagenet')

# make predictions
def predict_image(image_file):
    img = Image.open(image_file.stream) 
    img = img.resize((224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=1)[0] 

    predicted_label = decoded_preds[0][1] 
    return {'prediction': predicted_label} 

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        prediction = predict_image(file)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': f'Error predicting image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
