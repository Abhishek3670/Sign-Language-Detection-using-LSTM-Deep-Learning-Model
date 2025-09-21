"""
Real-time Sign Language Detection Web App using Streamlit.
Features: Webcam detection, video upload, model training, and performance metrics.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent))

from src.core.model import SignLanguageLSTM, DEFAULT_ACTIONS
from src.core.detector import RealTimeDetector, create_detector
from src.utils.preprocessing import MediaPipeProcessor
from config.config import *

# Configure logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'detection_running' not in st.session_state:
    st.session_state.detection_running = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'predictions' not in st.session_state:
    st.session_state.predictions = []


def load_css():
    """Load custom CSS for better styling."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B35;
        margin: 0.5rem 0;
    }
    
    .success-message {
        color: #28a745;
        background-color: #d4edda;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        color: #856404;
        background-color: #fff3cd;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
    
    .error-message {
        color: #721c24;
        background-color: #f8d7da;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)


def display_header():
    """Display the main application header."""
    st.markdown('<h1 class="main-header">🤟 Real-Time Sign Language Detection</h1>', 
                unsafe_allow_html=True)
    
    # Display GPU status prominently
    check_gpu_status()
    st.markdown("---")


def check_gpu_status():
    """Check and display GPU availability status."""
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        gpu_devices = tf.config.list_physical_devices('GPU')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if gpu_available:
                st.success(f"✅ GPU Available: {len(gpu_devices)} device(s)")
                for i, device in enumerate(gpu_devices):
                    st.info(f"GPU {i}: {device.name}")
            else:
                st.warning("⚠️ No GPU detected - Using CPU (slower performance)")
        
        with col2:
            # TensorFlow version info
            st.info(f"TensorFlow Version: {tf.__version__}")
            
            # Display performance tips
            if not gpu_available:
                with st.expander("📚 GPU Setup Guide"):
                    st.markdown("""
                    **To enable GPU acceleration:**
                    
                    1. **Check CUDA Compatibility:**
                       - Your system: CUDA 13.0 detected
                       - TensorFlow 2.15.0 supports CUDA 11.2-12.1
                    
                    2. **Install Compatible CUDA:**
                       - Download CUDA 12.1 from NVIDIA
                       - Install cuDNN 8.6+ for CUDA 12.1
                    
                    3. **Verify Installation:**
                       ```bash
                       python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
                       ```
                    
                    4. **Performance on CPU:**
                       - Reduce video resolution for better FPS
                       - Lower confidence threshold for faster detection
                       - Use shorter sequences for training
                    """)
            
    except Exception as e:
        st.error(f"Error checking GPU status: {e}")


def load_model_interface():
    """Interface for loading or creating the model."""
    st.subheader("🧠 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_path = st.text_input("Model Path", value=MODEL_PATH)
        actions_input = st.text_input("Actions (comma-separated)", 
                                    value=",".join(DEFAULT_ACTIONS))
        
        if st.button("Load/Create Model"):
            try:
                actions = np.array([action.strip() for action in actions_input.split(",")])
                
                with st.spinner("Loading model..."):
                    lstm_model = SignLanguageLSTM(actions)
                    
                    if os.path.exists(model_path):
                        lstm_model.load_model(model_path)
                        st.success(f"✅ Model loaded from {model_path}")
                    else:
                        lstm_model.build_model()
                        st.warning(f"⚠️ Model file not found. Created new model architecture.")
                    
                    st.session_state.detector = RealTimeDetector(lstm_model)
                    st.session_state.model_loaded = True
                    
                    # Display model info
                    st.info(f"Actions: {', '.join(actions)}")
                    st.info(f"Sequence Length: {lstm_model.sequence_length}")
                    st.info(f"Input Features: {lstm_model.keypoint_features}")
                    
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                logger.error(f"Model loading error: {e}")
    
    with col2:
        if st.session_state.model_loaded and st.session_state.detector:
            st.success("🎯 Model Ready for Detection")
            
            # Model summary
            if st.button("Show Model Summary"):
                if st.session_state.detector.model.model:
                    # Capture model summary
                    import io
                    import contextlib
                    
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        st.session_state.detector.model.model.summary()
                    summary = f.getvalue()
                    
                    st.text(summary)
        else:
            st.info("👆 Please load a model first")


def webcam_detection_interface():
    """Interface for real-time webcam detection."""
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first")
        return
    
    st.subheader("📹 Real-Time Webcam Detection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_threshold = st.slider("Confidence Threshold", 
                                        min_value=0.0, max_value=1.0, 
                                        value=CONFIDENCE_THRESHOLD, step=0.05)
    
    with col2:
        camera_index = st.selectbox("Camera Index", options=[0, 1, 2], index=0)
    
    with col3:
        show_landmarks = st.checkbox("Show Landmarks", value=True)
    
    # Update detector settings
    if st.session_state.detector:
        st.session_state.detector.confidence_threshold = confidence_threshold
    
    # Webcam controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_webcam = st.button("🎥 Start Webcam Detection", type="primary")
    
    with col2:
        stop_webcam = st.button("⏹️ Stop Detection")
    
    with col3:
        reset_stats = st.button("🔄 Reset Statistics")
    
    if reset_stats and st.session_state.detector:
        st.session_state.detector.reset_buffers()
        st.session_state.frame_count = 0
        st.session_state.predictions = []
        st.success("Statistics reset!")
    
    # Placeholder for video display
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    if start_webcam and st.session_state.detector:
        st.session_state.detection_running = True
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error(f"❌ Could not open camera {camera_index}")
            return
        
        try:
            while st.session_state.detection_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame, predicted_action, probabilities = \
                    st.session_state.detector.process_frame(frame)
                
                # Convert BGR to RGB for Streamlit
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                video_placeholder.image(processed_frame_rgb, channels="RGB", 
                                      use_column_width=True)
                
                # Update statistics
                st.session_state.frame_count += 1
                if predicted_action:
                    st.session_state.predictions.append({
                        'action': predicted_action,
                        'confidence': float(np.max(probabilities)),
                        'timestamp': time.time()
                    })
                
                # Display stats
                stats = st.session_state.detector.get_performance_stats()
                with stats_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Frames Processed", stats['frames_processed'])
                    with col2:
                        st.metric("FPS", f"{stats['average_fps']:.1f}")
                    with col3:
                        st.metric("Buffer Size", stats['sequence_buffer_size'])
                    with col4:
                        if predicted_action:
                            st.metric("Last Action", predicted_action)
                
                # Small delay to prevent overwhelming
                time.sleep(0.03)  # ~30 FPS
                
        except Exception as e:
            st.error(f"❌ Detection error: {e}")
            logger.error(f"Webcam detection error: {e}")
        
        finally:
            cap.release()
            st.session_state.detection_running = False
    
    if stop_webcam:
        st.session_state.detection_running = False


def video_upload_interface():
    """Interface for video file upload and processing."""
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first")
        return
    
    st.subheader("📁 Video File Processing")
    
    uploaded_file = st.file_uploader("Choose a video file", 
                                   type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider("Confidence Threshold (Video)", 
                                            min_value=0.0, max_value=1.0, 
                                            value=CONFIDENCE_THRESHOLD, step=0.05)
        
        with col2:
            if st.button("🎬 Process Video"):
                try:
                    # Update detector settings
                    st.session_state.detector.confidence_threshold = confidence_threshold
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process video
                    predictions = []
                    cap = cv2.VideoCapture(temp_video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    frame_count = 0
                    detected_actions = []
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process frame
                        processed_frame, predicted_action, probabilities = \
                            st.session_state.detector.process_frame(frame)
                        
                        if predicted_action:
                            detected_actions.append({
                                'frame': frame_count,
                                'action': predicted_action,
                                'confidence': float(np.max(probabilities)),
                                'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS)
                            })
                        
                        frame_count += 1
                        
                        # Update progress
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{total_frames}")
                    
                    cap.release()
                    
                    # Display results
                    st.success(f"✅ Video processed! Detected {len(detected_actions)} actions")
                    
                    if detected_actions:
                        # Create results dataframe
                        df = pd.DataFrame(detected_actions)
                        
                        # Display summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Actions", len(detected_actions))
                        
                        with col2:
                            unique_actions = df['action'].nunique()
                            st.metric("Unique Actions", int(unique_actions))
                        
                        with col3:
                            avg_confidence = df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                        
                        # Action timeline chart
                        fig = px.scatter(df, x='timestamp', y='action', 
                                       color='confidence', size='confidence',
                                       title="Action Detection Timeline",
                                       labels={'timestamp': 'Time (seconds)', 
                                             'action': 'Detected Action'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.subheader("Detection Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"detection_results_{int(time.time())}.csv",
                            mime="text/csv"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error processing video: {e}")
                    logger.error(f"Video processing error: {e}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)


def training_interface():
    """Interface for model training."""
    st.subheader("🏋️ Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_path = st.text_input("Training Data Path", value=DATA_PATH)
        epochs = st.number_input("Training Epochs", min_value=1, max_value=5000, 
                                value=DEFAULT_EPOCHS)
        
    with col2:
        save_path = st.text_input("Save Model Path", value=MODEL_PATH)
        test_size = st.slider("Test Size", min_value=0.05, max_value=0.3, 
                            value=TRAIN_TEST_SPLIT, step=0.05)
    
    if st.button("🚀 Start Training"):
        if not os.path.exists(data_path):
            st.error(f"❌ Data path not found: {data_path}")
            return
        
        try:
            # Create model
            actions = np.array([action.strip() for action in 
                              st.text_input("Training Actions", 
                                          value=",".join(DEFAULT_ACTIONS)).split(",")])
            
            lstm_model = SignLanguageLSTM(actions)
            lstm_model.build_model()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare data
            status_text.text("Preparing training data...")
            X_train, X_test, y_train, y_test = lstm_model.prepare_data(data_path)
            
            st.info(f"Training data shape: {X_train.shape}")
            st.info(f"Test data shape: {X_test.shape}")
            
            # Train model
            status_text.text("Training model...")
            history = lstm_model.train(X_train, y_train, epochs=epochs, verbose=0)
            
            # Save model
            lstm_model.save_model(save_path)
            
            st.success(f"✅ Training completed! Model saved to {save_path}")
            
            # Display training history
            if hasattr(history, 'history'):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Loss plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history.history['loss'], 
                                           name='Training Loss'))
                    if 'val_loss' in history.history:
                        fig.add_trace(go.Scatter(y=history.history['val_loss'], 
                                               name='Validation Loss'))
                    fig.update_layout(title="Training Loss", 
                                    xaxis_title="Epoch", 
                                    yaxis_title="Loss")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Accuracy plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history.history['categorical_accuracy'], 
                                           name='Training Accuracy'))
                    if 'val_categorical_accuracy' in history.history:
                        fig.add_trace(go.Scatter(y=history.history['val_categorical_accuracy'], 
                                               name='Validation Accuracy'))
                    fig.update_layout(title="Training Accuracy", 
                                    xaxis_title="Epoch", 
                                    yaxis_title="Accuracy")
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Training error: {e}")
            logger.error(f"Training error: {e}")


def performance_metrics():
    """Display performance metrics and statistics."""
    st.subheader("📊 Performance Metrics")
    
    if st.session_state.detector:
        stats = st.session_state.detector.get_performance_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Frames Processed", stats['frames_processed'])
        
        with col2:
            st.metric("Average FPS", f"{stats['average_fps']:.2f}")
        
        with col3:
            st.metric("Buffer Size", stats['sequence_buffer_size'])
        
        with col4:
            st.metric("Total Predictions", len(st.session_state.predictions))
        
        # Prediction history chart
        if st.session_state.predictions:
            df = pd.DataFrame(st.session_state.predictions)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Action distribution
                action_counts = df['action'].value_counts()
                fig = px.pie(values=action_counts.values, names=action_counts.index,
                           title="Action Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence over time
                fig = px.line(df, x=df.index, y='confidence', 
                            title="Confidence Over Time")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Load a model to see performance metrics")


def main():
    """Main application function."""
    load_css()
    display_header()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home",
        "🧠 Model Setup", 
        "📹 Webcam Detection",
        "📁 Video Processing",
        "🏋️ Training",
        "📊 Performance"
    ])
    
    # GPU status in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("System Status")
        check_gpu_status()
    
    # Main content based on selected page
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to Real-Time Sign Language Detection! 🤟
        
        This application uses LSTM deep learning models to detect sign language gestures 
        in real-time using your webcam or uploaded video files.
        
        ### Features:
        - 📹 **Real-time webcam detection**
        - 📁 **Video file processing** 
        - 🧠 **Model training and management**
        - 📊 **Performance metrics and analytics**
        - 💾 **Results export and download**
        
        ### Getting Started:
        1. Navigate to **Model Setup** to load or create a model
        2. Use **Webcam Detection** for real-time sign language recognition
        3. Upload videos in **Video Processing** for batch analysis
        4. Monitor performance in **Performance** metrics
        
        ### Supported Actions:
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("👋 **Hello**\nGreeting gesture")
        with col2:
            st.info("🙏 **Thanks**\nThank you gesture")
        with col3:
            st.info("❤️ **I Love You**\nI love you gesture")
    
    elif page == "🧠 Model Setup":
        load_model_interface()
    
    elif page == "📹 Webcam Detection":
        webcam_detection_interface()
    
    elif page == "📁 Video Processing":
        video_upload_interface()
    
    elif page == "🏋️ Training":
        training_interface()
    
    elif page == "📊 Performance":
        performance_metrics()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, TensorFlow, and MediaPipe | "
        "[GitHub Repository](https://github.com/Abhishek3670/Sign-Language-Detection-using-LSTM-Deep-Learning-Model)"
    )


if __name__ == "__main__":
    main()