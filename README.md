# RayDx Backend

A Flask-based REST API for chest X-ray pneumonia detection using deep learning. This backend service analyzes chest X-ray images and provides predictions with AI-generated radiological reports.

## 🏥 Project Overview

RayDx Backend is a medical AI application that:
- Analyzes chest X-ray images for pneumonia detection
- Uses a pre-trained MobileNetV2 model fine-tuned for binary classification
- Generates detailed radiological reports using xAI's Grok-3 model
- Provides confidence scores for predictions
- Includes comprehensive image validation and preprocessing

## 🚀 Features

- **Deep Learning Model**: MobileNetV2 architecture fine-tuned for chest X-ray analysis
- **AI-Generated Reports**: Integration with xAI's Grok-3 for professional radiological reports
- **Image Validation**: Automatic detection of X-ray images and quality checks
- **CORS Support**: Configured for frontend integration
- **Comprehensive Logging**: Detailed request and error logging
- **Health Checks**: Built-in health monitoring endpoints
- **Production Ready**: Optimized for deployment with Gunicorn

## 📋 Prerequisites

- Python 3.8+
- PyTorch (CPU/GPU/MPS support)
- xAI API key
- Virtual environment (recommended)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Backend
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   XAI_API_KEY=your_xai_api_key_here
   PORT=5000
   ```

## 🏃‍♂️ Quick Start

### Running the Development Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Production Deployment

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📊 Model Training

### Data Preprocessing

Before training, preprocess your dataset:

```bash
python preprocess.py --input raydx_dataset/ --output processed_dataset/
```

### Training the Model

```bash
python train_model.py
```

The trained model will be saved as `pneumonia_model.pth`.

## 🔌 API Endpoints

### Health Check
```http
GET /
```
Returns server status.

### Debug Route
```http
GET /debug
```
Returns debug information.

### Prediction Endpoint
```http
POST /predict
Content-Type: multipart/form-data
```

**Request Body:**
- `file`: Chest X-ray image (JPEG/PNG, minimum 100x100 pixels)

**Response:**
```json
{
  "prediction": "normal|pneumonia",
  "confidence": 95.67,
  "report": "Detailed radiological report..."
}
```

**Error Responses:**
```json
{
  "error": "Error message description"
}
```

## 📝 API Usage Examples

### Using cURL
```bash
curl -X POST \
  http://localhost:5000/predict \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@chest_xray.jpg'
```

### Using Python Requests
```python
import requests

url = "http://localhost:5000/predict"
files = {"file": open("chest_xray.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")
print(f"Report: {result['report']}")
```

### Using JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Prediction:', data.prediction);
  console.log('Confidence:', data.confidence);
  console.log('Report:', data.report);
});
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `XAI_API_KEY` | xAI API key for report generation | Required |
| `PORT` | Server port | 5000 |

### CORS Configuration

The API is configured to accept requests from:
- Development: All origins
- Production: `https://raydx-frontend.vercel.app`

## 📁 Project Structure

```
Backend/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── preprocess.py          # Data preprocessing script
├── pneumonia_model.pth    # Trained model weights
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
└── venv/                  # Virtual environment
```

## 🔍 Model Details

- **Architecture**: MobileNetV2
- **Input Size**: 224x224 pixels
- **Classes**: 2 (normal, pneumonia)
- **Preprocessing**: Resize, normalize, RGB conversion
- **Device Support**: CPU, CUDA, MPS (Apple Silicon)

## 🚨 Error Handling

The API includes comprehensive error handling for:
- Invalid file types
- Missing files
- Image size validation
- X-ray detection
- Model inference errors
- API rate limiting

## 📊 Logging

All API calls and errors are logged to:
- Console output
- `api_calls.log` file

Log format: `YYYY-MM-DD HH:MM:SS - LEVEL - MESSAGE`

## 🔒 Security Considerations

- CORS restrictions for production
- File type validation
- Image size limits
- Environment variable protection
- Input sanitization

## 🧪 Testing

### Health Check
```bash
curl http://localhost:5000/
```

### Debug Route
```bash
curl http://localhost:5000/debug
```

### Test with Sample Image
```bash
curl -X POST \
  http://localhost:5000/predict \
  -F 'file=@pneumonia1.jpeg'
```

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (if needed)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
- Check the logs in `api_calls.log`
- Verify your xAI API key is valid
- Ensure your image meets the requirements
- Check the health endpoint for server status

## 🔄 Version History

- **v1.0.0**: Initial release with MobileNetV2 model and xAI integration
- Basic pneumonia detection
- AI-generated radiological reports
- Comprehensive error handling and logging 