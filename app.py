from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# Load Models from the 'models' directory
cnn_tflite_model_path =("models/model.tflite")
svm_model = joblib.load("models/svm_kidney_model.pkl")
pca_model = joblib.load("models/pca_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=cnn_tflite_model_path)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Image Preprocessing
IMG_SIZE = (128, 128)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    # CNN Preprocessing (RGB Normalized)
    img_cnn = cv2.resize(img, IMG_SIZE) / 255.0
    img_cnn = np.expand_dims(img_cnn, axis=0).astype(np.float32)  # Ensure correct type

    # SVM Preprocessing (Grayscale & PCA)
    img_svm = cv2.resize(img, IMG_SIZE)
    img_svm = cv2.cvtColor(img_svm, cv2.COLOR_BGR2GRAY).flatten()
    img_svm_pca = pca_model.transform([img_svm])  # Apply PCA

    return img_cnn, img_svm_pca

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filepath = "temp.jpg"
    file.save(filepath)

    # Preprocess Image
    img_cnn, img_svm_pca = preprocess_image(filepath)
    os.remove(filepath)  # Delete temp file

    if img_cnn is None:
        return jsonify({"error": "Invalid image format"}), 400

    # TFLite CNN Prediction
    interpreter.set_tensor(input_details[0]['index'], img_cnn)
    interpreter.invoke()
    cnn_probs = interpreter.get_tensor(output_details[0]['index'])[0]

    cnn_pred_idx = np.argmax(cnn_probs)
    cnn_pred_label = label_encoder.inverse_transform([cnn_pred_idx])[0]
    cnn_confidence = float(cnn_probs[cnn_pred_idx])

    # SVM Prediction
    svm_probs = svm_model.predict_proba(img_svm_pca)[0]
    svm_pred_idx = np.argmax(svm_probs)
    svm_pred_label = label_encoder.inverse_transform([svm_pred_idx])[0]
    svm_confidence = float(svm_probs[svm_pred_idx])

    # Suggestions based on predictions
    suggestions = {
        "Stone": "Consult a urologist for further examination and possible treatment.",
        "Normal": "Your kidney appears normal, but regular check-ups are advised.",
        "NOT_KIDNEY_IMAGES": "Please upload a valid kidney image for proper diagnosis."
    }

    return jsonify({
        "cnn_prediction": cnn_pred_label,
        "cnn_confidence": cnn_confidence,
        "svm_prediction": svm_pred_label,
        "svm_confidence": svm_confidence,
        "suggestion": suggestions.get(cnn_pred_label, "No specific suggestion available.")
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
