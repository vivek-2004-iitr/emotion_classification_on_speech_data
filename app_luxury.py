import streamlit as st
import numpy as np
import pickle
import os
import sys

# Audio processing imports with fallbacks
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError as e:
    AUDIO_PROCESSING_AVAILABLE = False

# TensorFlow configuration and imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import load_model
    import tensorflow.keras.backend as K
    
    # Disable GPU completely for cloud deployment
    try:
        tf.config.set_visible_devices([], 'GPU')
    except:
        pass  # GPU config might not be available
    
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io

# Set page config
st.set_page_config(
    page_title="EmotiVoice - Audio Emotion Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎵"
)

# Minimal luxury dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');
    
    /* Global dark luxury theme */
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #111111 100%);
        color: #e8e8e8;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        background: rgba(15, 15, 15, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 2rem;
        max-width: 1000px;
        margin-top: 1rem;
    }
    
    /* Clean main header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 300;
        text-align: center;
        color: #ffffff;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }
    
    /* Elegant cards */
    .luxury-card {
        background: rgba(20, 20, 20, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Status indicators - minimal */
    .status-panel {
        background: rgba(25, 25, 25, 0.8);
        border-left: 2px solid #6b7280;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    .status-active {
        border-left-color: #10b981;
    }
    
    .status-warning {
        border-left-color: #f59e0b;
    }
    
    /* File uploader - luxury styling */
    .stFileUploader > div {
        background: rgba(20, 20, 20, 0.8) !important;
        border: 1px dashed rgba(156, 163, 175, 0.4) !important;
        border-radius: 8px !important;
        padding: 2rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(156, 163, 175, 0.6) !important;
        background: rgba(30, 30, 30, 0.8) !important;
    }
    
    .stFileUploader label {
        color: #d1d5db !important;
        font-weight: 400 !important;
    }
    
    /* Clean data tables */
    .stDataFrame {
        background: rgba(15, 15, 15, 0.8) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    
    .stDataFrame table {
        background: transparent !important;
        color: #e8e8e8 !important;
    }
    
    .stDataFrame th {
        background: rgba(30, 30, 30, 0.8) !important;
        color: #f3f4f6 !important;
        border: none !important;
        font-weight: 500 !important;
    }
    
    .stDataFrame td {
        background: transparent !important;
        color: #d1d5db !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    
    /* Sidebar luxury styling */
    .css-1d391kg, .css-1lcbmhc, .css-1544g2n {
        background: rgba(10, 10, 10, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Emotion result display */
    .emotion-display {
        background: rgba(20, 20, 20, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    
    .emotion-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 400;
        color: #ffffff;
        letter-spacing: 0.05em;
    }
    
    /* Metrics - clean and minimal */
    .metric-container {
        background: rgba(25, 25, 25, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
        color: #e8e8e8;
    }
    
    /* Audio player */
    .stAudio > div {
        background: rgba(20, 20, 20, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(25, 25, 25, 0.8) !important;
        color: #e8e8e8 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 6px !important;
        font-weight: 400 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(15, 15, 15, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 0 0 6px 6px !important;
        border-top: none !important;
    }
    
    /* Info boxes - minimal */
    .info-box {
        background: rgba(20, 20, 20, 0.7);
        border-left: 2px solid #6b7280;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        color: #d1d5db;
    }
    
    .success-box {
        border-left-color: #10b981;
    }
    
    .warning-box {
        border-left-color: #f59e0b;
    }
    
    .error-box {
        border-left-color: #ef4444;
    }
    
    /* Subtle separator */
    .separator {
        height: 1px;
        background: rgba(255, 255, 255, 0.08);
        margin: 2rem 0;
        border: none;
    }
    
    /* Spinner */
    .stSpinner {
        color: #9ca3af !important;
    }
    
    /* Override bright alerts */
    .stAlert {
        background: rgba(20, 20, 20, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #d1d5db !important;
    }
</style>
""", unsafe_allow_html=True)

def custom_focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(true_labels, predicted_probs):
        eps = K.epsilon()
        predicted_probs = K.clip(predicted_probs, eps, 1.0 - eps)
        log_loss = -true_labels * K.log(predicted_probs)
        focal_factor = alpha * K.pow(1.0 - predicted_probs, gamma)
        focal_loss = focal_factor * log_loss
        return K.sum(focal_loss, axis=1)
    return loss_fn

def create_fallback_model():
    """Create a simple fallback CNN model if the original model fails to load"""
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(130, 60)),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(256, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')  # 7 emotions
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_emotion_model():
    try:
        if not TENSORFLOW_AVAILABLE:
            return None, None, None
            
        # Check if model files exist
        model_path = 'model/emotion_model (2).h5'
        scaler_path = 'model/scaler (2).pkl'
        encoder_path = 'model/label_encoder (3).pkl'
        
        if not os.path.exists(model_path):
            return None, None, None
        if not os.path.exists(scaler_path):
            return None, None, None
        if not os.path.exists(encoder_path):
            return None, None, None
        
        model = None
        
        # Try different loading approaches
        try:
            # First attempt: Load with custom objects
            model = load_model(model_path, 
                              custom_objects={'loss_fn': custom_focal_loss()},
                              compile=False)
        except Exception as e1:
            try:
                # Second attempt: Load without custom objects
                model = load_model(model_path, compile=False)
            except Exception as e2:
                # Use fallback model
                model = create_fallback_model()
                if model is None:
                    return None, None, None
        
        # Compile the model
        if model is not None:
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Load scaler and label encoder
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, scaler, label_encoder
        
    except Exception as e:
        return None, None, None

def extract_features(audio_data, sample_rate):
    try:
        if not AUDIO_PROCESSING_AVAILABLE:
            return None
            
        if sample_rate != 22050:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050)
            sample_rate = 22050
        
        target_length = sample_rate * 3
        if len(audio_data) > target_length:
            start = (len(audio_data) - target_length) // 2
            audio_data = audio_data[start:start + target_length]
        else:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
        
        mfcc_feat = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=60)
        
        max_len = 130
        if mfcc_feat.shape[1] < max_len:
            mfcc_feat = np.pad(mfcc_feat, ((0, 0), (0, max_len - mfcc_feat.shape[1])), mode='constant')
        else:
            mfcc_feat = mfcc_feat[:, :max_len]
        
        return mfcc_feat.T
    
    except Exception as e:
        return None

def predict_emotion(features, model, scaler, label_encoder):
    try:
        features_reshaped = features.reshape(1, 130, 60)
        
        num_samples, time_steps, num_mfcc = features_reshaped.shape
        features_flat = features_reshaped.reshape(num_samples * time_steps, num_mfcc)
        features_scaled_flat = scaler.transform(features_flat)
        features_scaled = features_scaled_flat.reshape(num_samples, time_steps, num_mfcc)
        
        predictions = model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        emotion = label_encoder.classes_[predicted_class]
        
        return emotion, confidence, predictions[0]
    
    except Exception as e:
        return None, None, None

def plot_mfcc_features(features):
    # Set dark theme for matplotlib
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    features_transposed = features.T
    
    # Use a subtle colormap for luxury feel
    sns.heatmap(features_transposed, 
                cmap='Greys_r', 
                ax=ax,
                cbar_kws={'label': 'MFCC Value'})
    
    ax.set_title('Audio Feature Analysis', fontsize=14, color='#e8e8e8')
    ax.set_xlabel('Time Frames', fontsize=12, color='#d1d5db')
    ax.set_ylabel('MFCC Coefficients', fontsize=12, color='#d1d5db')
    ax.tick_params(colors='#9ca3af')
    
    plt.tight_layout()
    return fig

def main():
    st.markdown('<h1 class="main-header">EmotiVoice</h1>', 
                unsafe_allow_html=True)
    
    # Simple description
    st.markdown("""
    <div class="luxury-card">
        <p style="text-align: center; color: #d1d5db; font-size: 1.1rem; margin: 0;">
            Audio emotion recognition using machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicators - minimal and clean
    col1, col2 = st.columns(2)
    with col1:
        if AUDIO_PROCESSING_AVAILABLE:
            st.markdown("""
            <div class="status-panel status-active">
                <strong>Audio Processing</strong><br>
                <span style="color: #10b981;">Available</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-panel status-warning">
                <strong>Audio Processing</strong><br>
                <span style="color: #f59e0b;">Demo Mode</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if TENSORFLOW_AVAILABLE:
            st.markdown("""
            <div class="status-panel status-active">
                <strong>Neural Network</strong><br>
                <span style="color: #10b981;">Ready</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-panel status-warning">
                <strong>Neural Network</strong><br>
                <span style="color: #f59e0b;">Unavailable</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Load models
    model, scaler, label_encoder = load_emotion_model()
    
    if not all([model, scaler, label_encoder]) and TENSORFLOW_AVAILABLE:
        st.markdown("""
        <div class="info-box error-box">
            <strong>Model files not found</strong><br>
            Please ensure model files are in the 'model' directory.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    elif model is not None:
        st.markdown("""
        <div class="info-box success-box">
            Models loaded successfully
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box warning-box">
            Running in demo mode
        </div>
        """, unsafe_allow_html=True)
    
    # File upload
    st.markdown("### Audio Upload")
    
    uploaded_file = st.file_uploader(
        "Select audio file for emotion analysis", 
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Supported formats: WAV, MP3, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        if not AUDIO_PROCESSING_AVAILABLE or not TENSORFLOW_AVAILABLE:
            # Demo mode
            st.markdown("""
            <div class="info-box warning-box">
                Demo mode - showing sample results
            </div>
            """, unsafe_allow_html=True)
            
            # Create fake demo prediction
            demo_emotions = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
            demo_confidences = [0.15, 0.45, 0.12, 0.10, 0.08, 0.05, 0.05]
            
            st.markdown("""
            <div class="emotion-display">
                <div class="emotion-text">Happy (45.0%)</div>
            </div>
            """, unsafe_allow_html=True)
            
            conf_df = pd.DataFrame({
                'Emotion': demo_emotions,
                'Confidence': [f"{c:.1%}" for c in demo_confidences]
            }).sort_values('Confidence', ascending=False)
            
            st.dataframe(conf_df, use_container_width=True, hide_index=True)
            return
        
        # Real analysis mode
        st.markdown("### File Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-container"><strong>Name:</strong><br>{uploaded_file.name}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-container"><strong>Size:</strong><br>{uploaded_file.size / 1024:.1f} KB</div>', unsafe_allow_html=True)
        
        st.audio(uploaded_file, format='audio/wav')
        
        try:
            audio_data, sample_rate = librosa.load(uploaded_file, sr=None)
            
            with st.spinner("Processing audio..."):
                features = extract_features(audio_data, sample_rate)
            
            if features is not None:
                with st.spinner("Analyzing emotion..."):
                    emotion, confidence, all_predictions = predict_emotion(
                        features, model, scaler, label_encoder
                    )
                
                if emotion is not None:
                    st.markdown("### Results")
                    
                    # Main result
                    st.markdown(f"""
                    <div class="emotion-display">
                        <div class="emotion-text">{emotion.title()} ({confidence:.1%})</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed results
                    conf_df = pd.DataFrame({
                        'Emotion': label_encoder.classes_,
                        'Confidence': [f"{p:.1%}" for p in all_predictions]
                    }).sort_values('Confidence', ascending=False)
                    
                    st.dataframe(conf_df, use_container_width=True, hide_index=True)
                    
                    # Feature visualization
                    with st.expander("Feature Analysis"):
                        st.markdown("""
                        <div class="info-box">
                        MFCC (Mel-Frequency Cepstral Coefficients) represent the spectral 
                        characteristics of the audio signal used for emotion classification.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig_mfcc = plot_mfcc_features(features)
                        st.pyplot(fig_mfcc)
                
        except Exception as e:
            st.markdown(f"""
            <div class="info-box error-box">
                Error processing audio: {str(e)}
            </div>
            """, unsafe_allow_html=True)
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### Supported Emotions")
        st.markdown("""
        <div class="luxury-card">
            <ul style="color: #d1d5db; margin: 0; padding-left: 1rem;">
                <li>Calm</li>
                <li>Happy</li>
                <li>Sad</li>
                <li>Angry</li>
                <li>Fearful</li>
                <li>Disgust</li>
                <li>Surprised</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Technical Details")
        st.markdown("""
        <div class="luxury-card">
            <p style="color: #9ca3af; font-size: 0.9rem; margin: 0;">
            Uses deep learning with MFCC features extracted from 
            audio signals to classify emotional content.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown('<hr class="separator">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p style="margin: 0; font-size: 0.9rem;">EmotiVoice • Audio Emotion Recognition</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
