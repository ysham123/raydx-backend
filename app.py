import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from dotenv import load_dotenv
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_calls.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
XAI_API_KEY = os.getenv('XAI_API_KEY')
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY is not set in .env file")

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for requests from Vercel frontend
CORS(app, resources={r"/predict": {"origins": ["http://localhost:3000", "https://raydx-frontend.vercel.app"]}})

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the MobileNetV2 model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=2)
model_path = "pneumonia_model.pth"  
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully")

# Define labels
labels = ["normal", "pneumonia"]

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

def is_likely_xray(image):
    """Basic heuristic to check if the image is likely an X-ray (grayscale dominance)."""
    try:
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # Grayscale image
            return True
        if len(img_array.shape) == 3:  # RGB image
            # Check if the image is effectively grayscale by comparing channels
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            if np.allclose(r, g, atol=10) and np.allclose(g, b, atol=10):
                return True
        return False
    except Exception as e:
        logger.error(f"Error in is_likely_xray: {str(e)}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({"error": "Invalid file type. Please upload a JPEG or PNG image."}), 400
    
    try:
        # Read file bytes once and store
        image_bytes = file.read()
        
        # Create PIL Image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Validate image dimensions (minimum size)
        width, height = image.size
        if width < 100 or height < 100:
            return jsonify({"error": "Image is too small. Minimum dimensions are 100x100 pixels."}), 400
        
        # Check if the image is likely an X-ray
        if not is_likely_xray(image):
            return jsonify({"error": "Uploaded image does not appear to be an X-ray. Please upload a chest X-ray image."}), 400
        
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_label = labels[predicted.item()]
            confidence_value = confidence.item() * 100
        
        # Generate prompt and send to xAI API
        prompt = (
            f"A chest X-ray was analyzed using a deep learning model, predicting {predicted_label} "
            f"with {confidence_value:.2f}% confidence. "
            "Generate a concise radiological report for an experienced radiologist, "
            "focusing solely on the predicted radiographic findings in a single, articulate sentence. "
            "Provide an in-depth description of the findings using precise, technical medical terminology, "
            "while avoiding any additional metadata, formatting, symbols like asterisks, "
            "and excluding differential diagnoses or clinical recommendations."
        )
        
        logger.info(f"Sending prompt to xAI API: {prompt}")
        
        try:
            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {XAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'grok-3',
                    'messages': [
                        {'role': 'system', 'content': 'You are a medical AI assistant specializing in radiology.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    'max_tokens': 150,
                    'temperature': 0.7
                }
            )
            response.raise_for_status()
            api_response = response.json()
            report = api_response['choices'][0]['message']['content']
            logger.info(f"xAI API response: {api_response}")
        except requests.RequestException as e:
            report = f"Error generating report: {str(e)}"
            logger.error(f"xAI API error: {str(e)}")
        
        # Return prediction, confidence, report, and image as hex
        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence_value,
            "report": report,
            "image": image_bytes.hex()
        })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": "Failed to process the image. Please ensure it is a valid chest X-ray image."}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)