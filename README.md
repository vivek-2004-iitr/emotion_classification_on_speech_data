# Audio Emotion Recognition App

An intelligent Streamlit web application that analyzes audio files to detect emotional content using deep learning. The system can identify seven different emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, and Disgust.

## 🌐 Live Demo

� **[Try the app on Streamlit Cloud](https://emotionclassificationonspeechdata-hwgfvwrm32azfm6pqydvjm.streamlit.app/)**

*Note: The cloud version runs in demo mode due to audio library constraints. For full functionality, please run locally.*

## �🚀 Features

- **Real-time Emotion Detection**: Upload audio files and get instant emotion predictions
- **Multiple Audio Formats**: Supports WAV, MP3, FLAC, and M4A files (local version)
- **Visual Analysis**: Interactive MFCC feature visualization and confidence scoring
- **User-friendly Interface**: Clean, modern UI with detailed explanations
- **High Accuracy**: Deep CNN model with SE-Blocks for robust emotion recognition
- **Demo Mode**: Cloud deployment shows sample predictions without audio processing

## 🎯 Supported Emotions

- Neutral
- Calm  
- Happy
- Sad
- Angry
- Fearful
- Disgust

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: TensorFlow/Keras
- **Audio Processing**: Librosa (local version)
- **Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients)
- **Visualization**: Matplotlib, Seaborn

## 📊 Model Architecture

- Deep Convolutional Neural Network with SE-Blocks
- 60 MFCC coefficients as input features
- Processes 3-second audio segments at 22,050 Hz sample rate
- Custom focal loss for handling class imbalance

## 🚀 Quick Start

### Local Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd emotion_classification_on_speechdata
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

This app is ready for deployment on Streamlit Cloud. Simply:

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy with one click!

## 📁 Project Structure

```
emotion_classification_on_speechdata/
├── app.py                 # Main Streamlit application
├── model/                 # Pre-trained model files
│   ├── emotion_model (2).h5
│   ├── label_encoder (3).pkl
│   └── scaler (2).pkl
├── requirements.txt       # Python dependencies
├── packages.txt          # System packages for Streamlit Cloud
├── runtime.txt           # Python version specification
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 🎵 Usage

1. **Upload Audio**: Choose an audio file in supported formats
2. **View Analysis**: See file information and audio player
3. **Get Predictions**: View detected emotion with confidence scores
4. **Explore Features**: Examine MFCC visualizations and audio characteristics

## 📋 Requirements

See `requirements.txt` for detailed Python package requirements.

## 🔧 Configuration

The app includes optimized configuration for Streamlit Cloud deployment:
- Maximum upload size: 200MB
- Custom theme with professional styling
- Efficient caching for model loading

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.
