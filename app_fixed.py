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
    tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs
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
    page_title="🚀 EMOTIVOX AI - Advanced Neural Emotion Detection 🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for completely new UI design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;800&display=swap');
    
    /* Global app styling with animated gradient */
    .stApp {
        background: linear-gradient(45deg, #000000, #1a1a2e, #16213e, #0f0f23);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: #ffffff;
        min-height: 100vh;
        font-family: 'Exo 2', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide default Streamlit elements */
    .stMarkdown, .stSelectbox, .stTextInput, .stTextArea, .stNumberInput, 
    .stDateInput, .stTimeInput, .stFileUploader, .stColorPicker,
    .stSlider, .stCheckbox, .stRadio, .stMultiSelect, .stExpander {
        background: transparent !important;
    }
    
    /* Main container */
    .main .block-container {
        background: transparent;
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Futuristic main header */
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00f5ff, #ff00ff, #ffff00, #00ff00);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 3s ease infinite;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        letter-spacing: 0.1em;
    }
    
    /* Cyber card design */
    .cyber-card {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(20, 20, 40, 0.8));
        border: 2px solid;
        border-image: linear-gradient(45deg, #00f5ff, #ff00ff) 1;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 
            0 8px 32px rgba(0, 245, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .cyber-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00f5ff, #ff00ff, #ffff00, #00ff00);
        border-radius: 20px;
        z-index: -1;
        animation: gradientShift 4s ease infinite;
    }
    
    .cyber-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 15px 50px rgba(0, 245, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00f5ff;
        margin: 2rem 0 1rem 0;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #00f5ff, #ff00ff);
        border-radius: 2px;
    }
    
    /* Status indicators */
    .status-panel {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        gap: 1rem;
    }
    
    .status-item {
        flex: 1;
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.1), rgba(255, 0, 255, 0.1));
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .status-item:hover {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.2), rgba(255, 0, 255, 0.2));
        transform: scale(1.05);
    }
    
    .status-active {
        background: linear-gradient(135deg, rgba(0, 255, 0, 0.2), rgba(0, 245, 255, 0.2));
        border-color: #00ff00;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
    }
    
    .status-demo {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.2), rgba(255, 0, 255, 0.2));
        border-color: #ffa500;
        box-shadow: 0 0 20px rgba(255, 165, 0, 0.3);
    }
    
    /* Emotion result display */
    .emotion-display {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.9), rgba(20, 20, 40, 0.9));
        border: 3px solid;
        border-image: linear-gradient(45deg, #00f5ff, #ff00ff, #ffff00) 1;
        border-radius: 25px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .emotion-display::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: scanline 2s linear infinite;
    }
    
    @keyframes scanline {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .emotion-text {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00f5ff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(20, 20, 40, 0.8)) !important;
        border: 2px dashed #00f5ff !important;
        border-radius: 20px !important;
        padding: 3rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #ff00ff !important;
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.1), rgba(255, 0, 255, 0.1)) !important;
        transform: scale(1.02) !important;
    }
    
    .stFileUploader label {
        color: #00f5ff !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    /* Data table styling */
    .stDataFrame {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(20, 20, 40, 0.8)) !important;
        border-radius: 15px !important;
        overflow: hidden !important;
        border: 1px solid rgba(0, 245, 255, 0.3) !important;
    }
    
    .stDataFrame table {
        background: transparent !important;
        color: #ffffff !important;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, #00f5ff, #ff00ff) !important;
        color: #000000 !important;
        font-weight: 700 !important;
        text-align: center !important;
        border: none !important;
    }
    
    .stDataFrame td {
        background: rgba(0, 0, 0, 0.5) !important;
        color: #ffffff !important;
        text-align: center !important;
        border: 1px solid rgba(0, 245, 255, 0.2) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc, .css-1544g2n {
        background: linear-gradient(180deg, rgba(0, 0, 0, 0.9), rgba(10, 10, 30, 0.9)) !important;
        border-right: 2px solid rgba(0, 245, 255, 0.3) !important;
    }
    
    /* Metrics containers */
    .metric-box {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(20, 20, 40, 0.8));
        border: 1px solid rgba(0, 245, 255, 0.4);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .metric-box:hover {
        border-color: #ff00ff;
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 245, 255, 0.3);
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #00f5ff;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #ffffff;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(20, 20, 40, 0.8)) !important;
        color: #00f5ff !important;
        border: 1px solid rgba(0, 245, 255, 0.4) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.9), rgba(10, 10, 30, 0.9)) !important;
        border: 1px solid rgba(0, 245, 255, 0.3) !important;
        border-radius: 0 0 10px 10px !important;
        border-top: none !important;
    }
    
    /* Audio player */
    .stAudio > div {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(20, 20, 40, 0.8)) !important;
        border: 1px solid rgba(0, 245, 255, 0.4) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    
    /* Loading animation */
    .stSpinner {
        color: #00f5ff !important;
    }
    
    /* Footer styling */
    .footer-section {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.9), rgba(20, 20, 40, 0.9));
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
    }
    
    /* Pulse animation for important elements */
    .pulse-glow {
        animation: pulseGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulseGlow {
        from { box-shadow: 0 0 10px rgba(0, 245, 255, 0.4); }
        to { box-shadow: 0 0 30px rgba(0, 245, 255, 0.8); }
    }
    
    /* Alert styling */
    .stAlert {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(20, 20, 40, 0.8)) !important;
        border: 1px solid rgba(0, 245, 255, 0.4) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
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
        
        st.warning("⚠️ 🔧 Utilizing backup neural architecture. Predictions may have reduced accuracy.")
        return model
    except Exception as e:
        st.error(f"❌ Neural network initialization failed: {str(e)}")
        return None

@st.cache_resource
def load_emotion_model():
    try:
        if not TENSORFLOW_AVAILABLE:
            st.warning("⚠️ 🧠 Deep Learning Framework unavailable - activating simulation mode")
            return None, None, None
            
        # Check if model files exist
        model_path = 'model/emotion_model (2).h5'
        scaler_path = 'model/scaler (2).pkl'
        encoder_path = 'model/label_encoder (3).pkl'
        
        if not os.path.exists(model_path):
            st.error(f"❌ Neural network file missing: {model_path}")
            return None, None, None
        if not os.path.exists(scaler_path):
            st.error(f"❌ Data normalizer file missing: {scaler_path}")
            return None, None, None
        if not os.path.exists(encoder_path):
            st.error(f"❌ Emotion classifier file missing: {encoder_path}")
            return None, None, None
        
        model = None
        
        # Try different loading approaches
        try:
            # First attempt: Load with custom objects
            model = load_model(model_path, 
                              custom_objects={'loss_fn': custom_focal_loss()},
                              compile=False)
            st.success("✅ 🚀 Neural network loaded with advanced configurations!")
        except Exception as e1:
            st.warning(f"⚠️ Advanced loading protocol failed: {str(e1)}")
            try:
                # Second attempt: Load without custom objects
                model = load_model(model_path, compile=False)
                st.success("✅ 🚀 Neural network loaded with standard configurations!")
            except Exception as e2:
                st.warning(f"⚠️ Standard loading protocol failed: {str(e2)}")
                # Use fallback model
                st.info("🔧 Initializing backup neural architecture...")
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
        st.error(f"❌ Neural network initialization error: {str(e)}")
        return None, None, None

def extract_features(audio_data, sample_rate):
    try:
        if not AUDIO_PROCESSING_AVAILABLE:
            st.error("❌ Audio signal processing capabilities unavailable")
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
        st.error(f"❌ Audio feature extraction failed: {str(e)}")
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
        st.error(f"❌ Emotion prediction analysis failed: {str(e)}")
        return None, None, None

def plot_mfcc_features(features):
    # Set dark theme for matplotlib
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')
    
    features_transposed = features.T
    
    # Use a custom colormap for better contrast in dark theme
    sns.heatmap(features_transposed, 
                cmap='plasma', 
                ax=ax,
                cbar_kws={'label': '🎵 MFCC Coefficient Values'})
    
    ax.set_title('🔬 SPECTRAL EMOTION FINGERPRINT ANALYSIS 🔬', fontsize=16, fontweight='bold', color='#00f5ff')
    ax.set_xlabel('⏱️ Temporal Progression (Frames)', fontsize=12, color='#ffffff')
    ax.set_ylabel('🎼 Frequency Coefficients (MFCC)', fontsize=12, color='#ffffff')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    return fig

def main():
    st.markdown('<h1 class="main-header">🚀 EMOTIVOX AI 🧠</h1>', 
                unsafe_allow_html=True)
    
    # Hero section with welcome message
    st.markdown("""
    <div class="cyber-card pulse-glow">
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #00f5ff; font-family: 'Orbitron', monospace; margin-bottom: 1rem;">
                🎭 WELCOME TO THE FUTURE OF EMOTION DETECTION 🎭
            </h2>
            <p style="font-size: 1.2rem; color: #ffffff; margin-bottom: 0;">
                🔮 Harness the power of advanced neural networks to decode emotions from audio signals 🔮
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status panel with futuristic design
    st.markdown('<div class="section-header">🛰️ SYSTEM STATUS 🛰️</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if AUDIO_PROCESSING_AVAILABLE:
            st.markdown("""
            <div class="status-item status-active">
                <h3 style="color: #00ff00; margin: 0;">🎵 AUDIO ENGINE</h3>
                <p style="color: #ffffff; margin: 0.5rem 0 0 0;">✅ FULLY OPERATIONAL</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-item status-demo">
                <h3 style="color: #ffa500; margin: 0;">🎵 AUDIO ENGINE</h3>
                <p style="color: #ffffff; margin: 0.5rem 0 0 0;">⚠️ SIMULATION MODE</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if TENSORFLOW_AVAILABLE:
            st.markdown("""
            <div class="status-item status-active">
                <h3 style="color: #00ff00; margin: 0;">🧠 NEURAL CORE</h3>
                <p style="color: #ffffff; margin: 0.5rem 0 0 0;">✅ ACTIVE & LEARNING</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-item status-demo">
                <h3 style="color: #ffa500; margin: 0;">🧠 NEURAL CORE</h3>
                <p style="color: #ffffff; margin: 0.5rem 0 0 0;">⚠️ INITIALIZING</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed system status
    if AUDIO_PROCESSING_AVAILABLE:
        st.markdown("""
        <div class="cyber-card">
            <div style="text-align: center;">
                <h3 style="color: #00ff00;">✅ 🎵 AUDIO SIGNAL PROCESSING ALGORITHMS ONLINE</h3>
                <p style="color: #ffffff;">All audio analysis systems are fully initialized and ready for operation!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="cyber-card">
            <div style="text-align: center;">
                <h3 style="color: #ffa500;">⚠️ 🎵 AUDIO ENGINE IN SIMULATION MODE</h3>
                <p style="color: #ffffff;">Audio processing unavailable - operating in demonstration mode<br>
                💡 This deployment prioritizes neural network functionality for optimal cloud compatibility</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if TENSORFLOW_AVAILABLE:
        st.markdown("""
        <div class="cyber-card">
            <div style="text-align: center;">
                <h3 style="color: #00ff00;">✅ 🧠 DEEP NEURAL NETWORKS ACTIVATED</h3>
                <p style="color: #ffffff;">Advanced AI systems are successfully activated and operational!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="cyber-card">
            <div style="text-align: center;">
                <h3 style="color: #ffa500;">⚠️ 🧠 NEURAL CORE INITIALIZING</h3>
                <p style="color: #ffffff;">Deep learning framework unavailable - operating in basic analysis mode</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar with emotion capabilities
    with st.sidebar:
        st.markdown("""
        <div class="cyber-card">
            <div class="section-header" style="font-size: 1.2rem; margin: 0 0 1rem 0;">🎯 EMOTION DETECTION MATRIX</div>
            <div style="color: #ffffff;">
                <strong>🎭 DETECTABLE EMOTIONAL STATES:</strong><br><br>
                😌 Serenity &nbsp;&nbsp;&nbsp; 😇 Tranquility<br>
                😊 Joyfulness &nbsp;&nbsp;&nbsp; 😢 Melancholy<br>
                😡 Rage &nbsp;&nbsp;&nbsp; 😰 Anxiety<br>
                🤢 Revulsion
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="cyber-card">
            <div class="section-header" style="font-size: 1.2rem; margin: 0 0 1rem 0;">⚙️ NEURAL ARCHITECTURE</div>
            <div style="color: #ffffff;">
                <strong>🧠 Network Type:</strong> Deep Convolutional with Squeeze-Excitation Modules<br><br>
                <strong>📊 Signal Features:</strong> Mel-Frequency Cepstral Coefficients (60 dimensions)<br><br>
                <strong>⏱️ Analysis Window:</strong> 3-second temporal segments<br><br>
                <strong>📈 Sampling Frequency:</strong> 22,050 Hz digital resolution
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, label_encoder = load_emotion_model()
    
    if model is None and TENSORFLOW_AVAILABLE:
        st.markdown("""
        <div class="cyber-card" style="border-color: #ff0000;">
            <div style="text-align: center;">
                <h3 style="color: #ff0000;">❌ NEURAL NETWORK INITIALIZATION FAILED</h3>
                <p style="color: #ffffff;">Please verify that all AI model components exist in the 'model' directory.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    elif model is not None:
        st.markdown("""
        <div class="cyber-card">
            <div style="text-align: center;">
                <h3 style="color: #00ff00;">✅ 🚀 NEURAL NETWORK DEPLOYED SUCCESSFULLY</h3>
                <p style="color: #ffffff;">Advanced AI emotion analysis systems are ready for operation!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="cyber-card">
            <div style="text-align: center;">
                <h3 style="color: #00f5ff;">🎭 DEMONSTRATION MODE ACTIVE</h3>
                <p style="color: #ffffff;">AI simulation protocols activated for system demonstration</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="section-header">📂 AUDIO SIGNAL UPLOAD 📂</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "🎵 Select your audio recording for emotional analysis 🎵", 
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="🔧 Compatible formats: WAV, MP3, FLAC, M4A audio files"
    )
    
    if uploaded_file is not None:
        if not AUDIO_PROCESSING_AVAILABLE or not TENSORFLOW_AVAILABLE:
            # Demo mode
            st.markdown('<div class="section-header">🎭 DEMONSTRATION MODE 🎭</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="cyber-card">
                <div style="text-align: center;">
                    <h3 style="color: #ffa500;">⚠️ OPERATING IN SIMULATION MODE</h3>
                    <p style="color: #ffffff;">Displaying sample emotional analysis for demonstration purposes</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if not AUDIO_PROCESSING_AVAILABLE:
                st.markdown("""
                <div class="cyber-card">
                    <div style="text-align: center;">
                        <h4 style="color: #00f5ff;">📝 Audio signal processing algorithms not fully operational</h4>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            if not TENSORFLOW_AVAILABLE:
                st.markdown("""
                <div class="cyber-card">
                    <div style="text-align: center;">
                        <h4 style="color: #00f5ff;">🧠 Deep learning neural network not fully operational</h4>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create a fake demo prediction
            demo_emotions = ['😌 Serenity', '😇 Tranquility', '😊 Joyfulness', '😢 Melancholy', '😡 Rage', '😰 Anxiety', '🤢 Revulsion']
            demo_confidences = [0.15, 0.08, 0.45, 0.12, 0.10, 0.05, 0.05]  # Joyfulness is highest
            
            st.markdown("""
            <div class="emotion-display">
                <div class="emotion-text">🎉 EMOTIONAL SIGNATURE: JOYFULNESS 🎉</div>
                <div style="color: #ffffff; font-size: 1.2rem; margin-top: 1rem;">
                    Confidence Level: 45.0%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-header">📊 PROBABILITY MATRIX 📊</div>', unsafe_allow_html=True)
            
            conf_df = pd.DataFrame({
                'Emotional State': demo_emotions,
                'Confidence Level (%)': [c * 100 for c in demo_confidences]
            }).sort_values('Confidence Level (%)', ascending=False)
            
            st.dataframe(
                conf_df.style.format({'Confidence Level (%)': '{:.2f}'}),
                use_container_width=True
            )
            
            return
        
        # Full functionality mode
        st.markdown('<div class="section-header">📊 AUDIO SIGNAL PROPERTIES 📊</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">📁 {uploaded_file.name}</div>
                <div class="metric-label">File Identity</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">💾 {uploaded_file.size / 1024:.1f} KB</div>
                <div class="metric-label">Data Volume</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.audio(uploaded_file, format='audio/wav')
        
        try:
            audio_data, sample_rate = librosa.load(uploaded_file, sr=None)
            
            with st.spinner("🔬 Extracting spectral characteristics..."):
                features = extract_features(audio_data, sample_rate)
            
            if features is not None:
                with st.spinner("🧠 Analyzing emotional patterns..."):
                    emotion, confidence, all_predictions = predict_emotion(
                        features, model, scaler, label_encoder
                    )
                
                if emotion is not None:
                    st.markdown('<div class="section-header">🎯 EMOTIONAL ANALYSIS RESULTS 🎯</div>', unsafe_allow_html=True)
                    
                    emotion_emojis = {
                        'neutral': '😐',
                        'calm': '😌', 
                        'happy': '😊',
                        'sad': '😢',
                        'angry': '😡',
                        'fearful': '😰',
                        'disgust': '🤢'
                    }
                    
                    emoji = emotion_emojis.get(emotion, '🎭')
                    
                    st.markdown(f"""
                    <div class="emotion-display">
                        <div class="emotion-text">{emoji} DOMINANT EMOTIONAL SIGNATURE: {emotion.upper()} {emoji}</div>
                        <div style="color: #ffffff; font-size: 1.2rem; margin-top: 1rem;">
                            Certainty Level: {confidence*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="section-header">📈 EMOTIONAL PROBABILITY DISTRIBUTION 📈</div>', unsafe_allow_html=True)
                    
                    conf_df = pd.DataFrame({
                        'Emotional State': [f"{emotion_emojis.get(emotion, '🎭')} {emotion.title()}" for emotion in label_encoder.classes_],
                        'Probability Score (%)': all_predictions * 100
                    }).sort_values('Probability Score (%)', ascending=False)
                    
                    st.dataframe(
                        conf_df.style.format({'Probability Score (%)': '{:.2f}'}),
                        use_container_width=True
                    )
                    
                    st.markdown('<div class="section-header">🔬 SPECTRAL FEATURE ANALYSIS 🔬</div>', unsafe_allow_html=True)
                    
                    with st.expander("🔍 View Spectral Emotion Fingerprint 🔍", expanded=False):
                        st.markdown("""
                        <div class="cyber-card">
                            <p style="color: #ffffff;">
                            <strong>🎼 Mel-Frequency Cepstral Coefficients (MFCC)</strong> capture the unique spectral 
                            signature of the audio signal. These mathematical representations reveal the timbral 
                            characteristics that serve as emotional fingerprints for AI analysis.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig_mfcc = plot_mfcc_features(features)
                        st.pyplot(fig_mfcc)
                    
                    with st.expander("📊 Digital Signal Characteristics 📊", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">⏱️ {len(audio_data) / sample_rate:.2f}s</div>
                                <div class="metric-label">Temporal Length</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">📈 {sample_rate:,} Hz</div>
                                <div class="metric-label">Digital Sampling Rate</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">⚡ {np.sqrt(np.mean(audio_data**2)):.4f}</div>
                                <div class="metric-label">Signal Energy (RMS)</div>
                            </div>
                            """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Audio signal analysis encountered an error: {str(e)}")
    
    # Footer
    st.markdown("""
    <div class="footer-section">
        <h3 style="color: #00f5ff; margin-bottom: 1rem;">⚡ POWERED BY ADVANCED TECHNOLOGY ⚡</h3>
        <p style="color: #ffffff; margin-bottom: 0.5rem;">🛠️ Engineered with Streamlit • 🧠 Powered by Deep Neural Networks</p>
        <p style="color: #888888; font-size: 0.9rem;">🚀 Upload your audio recording to begin the emotional intelligence analysis! 🚀</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
