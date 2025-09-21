"""
LSTM model for sign language detection.
Extracted from notebook with training and inference capabilities.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguageLSTM:
    """LSTM model for sign language detection."""
    
    def __init__(self, actions: np.ndarray, sequence_length: int = 30, 
                 keypoint_features: int = 1662):
        """
        Initialize the LSTM model.
        
        Args:
            actions: Array of action labels
            sequence_length: Number of frames in sequence
            keypoint_features: Number of keypoint features per frame
        """
        self.actions = actions
        self.sequence_length = sequence_length
        self.keypoint_features = keypoint_features
        self.num_actions = len(actions)
        self.model = None
        self.label_map = {label: num for num, label in enumerate(actions)}
        
        # Robust GPU detection and configuration
        self.gpu_available = self._configure_gpu_support()
        logger.info(f"GPU Available: {self.gpu_available}")
        
        if self.gpu_available:
            self._enable_mixed_precision()
    
    def _configure_gpu_support(self) -> bool:
        """
        Robust GPU configuration with multiple detection methods.
        
        Returns:
            True if GPU is available and configured, False otherwise
        """
        try:
            # Method 1: Check physical devices
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logger.info("No GPU devices found via list_physical_devices")
                return False
            
            # Method 2: Check CUDA build
            if hasattr(tf.test, 'is_built_with_cuda'):
                cuda_build = tf.test.is_built_with_cuda()
                logger.info(f"TensorFlow built with CUDA: {cuda_build}")
                if not cuda_build:
                    logger.warning("TensorFlow not built with CUDA support")
                    return False
            
            # Method 3: Test GPU availability
            if hasattr(tf.test, 'is_gpu_available'):
                gpu_test = tf.test.is_gpu_available()
                logger.info(f"GPU test availability: {gpu_test}")
            
            # Configure GPU memory growth
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU memory growth configured for: {gpu.name}")
                except Exception as e:
                    logger.warning(f"Failed to configure GPU {gpu.name}: {e}")
            
            logger.info(f"Successfully configured {len(gpus)} GPU(s)")
            return True
            
        except Exception as e:
            logger.warning(f"GPU configuration failed: {e}")
            return False
    
    def _enable_mixed_precision(self):
        """
        Enable mixed precision for improved GPU performance.
        """
        try:
            # Enable mixed precision policy for better GPU performance
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled for improved GPU performance")
        except Exception as e:
            logger.warning(f"Failed to enable mixed precision: {e}")
    
    def build_model(self) -> Sequential:
        """
        Build the LSTM model architecture based on notebook implementation.
        
        Returns:
            Compiled Sequential model
        """
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', 
                 input_shape=(self.sequence_length, self.keypoint_features)),
            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.num_actions, activation='softmax')
        ])
        
        model.compile(optimizer='Adam', loss='categorical_crossentropy', 
                     metrics=['categorical_accuracy'])
        
        self.model = model
        logger.info(f"Model built with input shape: ({self.sequence_length}, {self.keypoint_features})")
        logger.info(f"Number of actions: {self.num_actions}")
        
        return model
    
    def prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, 
                                                   np.ndarray, np.ndarray]:
        """
        Prepare training data from stored numpy files.
        
        Args:
            data_path: Path to the MP_Data directory
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        sequences, labels = [], []
        
        for action in self.actions:
            action_path = os.path.join(data_path, action)
            if not os.path.exists(action_path):
                logger.warning(f"Action directory not found: {action_path}")
                continue
            
            # Get all sequence directories for this action
            sequence_dirs = [d for d in os.listdir(action_path) 
                           if os.path.isdir(os.path.join(action_path, d))]
            
            for sequence_dir in sequence_dirs:
                sequence_path = os.path.join(action_path, sequence_dir)
                window = []
                
                # Load sequence frames
                for frame_num in range(self.sequence_length):
                    frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                    if os.path.exists(frame_path):
                        frame_data = np.load(frame_path)
                        window.append(frame_data)
                    else:
                        logger.warning(f"Frame not found: {frame_path}")
                        # Use zeros if frame is missing
                        window.append(np.zeros(self.keypoint_features))
                
                if len(window) == self.sequence_length:
                    sequences.append(window)
                    labels.append(self.label_map[action])
        
        if not sequences:
            raise ValueError(f"No training data found in {data_path}")
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, 
                                                            random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 100, verbose: int = 1) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            epochs: Number of training epochs
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Set up TensorBoard callback
        log_dir = os.path.join('Logs')
        os.makedirs(log_dir, exist_ok=True)
        tb_callback = TensorBoard(log_dir=log_dir)
        
        logger.info(f"Starting training for {epochs} epochs...")
        history = self.model.fit(X_train, y_train, epochs=epochs, 
                                callbacks=[tb_callback], verbose=verbose)
        
        return history
    
    def save_model(self, filepath: str = 'action.h5'):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'action.h5'):
        """
        Load a pre-trained model.
        
        Args:
            filepath: Path to the saved model
        """
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        else:
            logger.warning(f"Model file not found: {filepath}")
            logger.info("Building new model...")
            self.build_model()
    
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Make prediction on a single sequence.
        
        Args:
            sequence: Input sequence of shape (sequence_length, keypoint_features)
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # Ensure correct input shape
        if sequence.shape != (self.sequence_length, self.keypoint_features):
            raise ValueError(f"Expected shape ({self.sequence_length}, {self.keypoint_features}), "
                           f"got {sequence.shape}")
        
        # Add batch dimension and predict
        prediction = self.model.predict(np.expand_dims(sequence, axis=0), verbose=0)
        return prediction[0]
    
    def predict_action(self, sequence: np.ndarray, threshold: float = 0.8) -> Optional[str]:
        """
        Predict the most likely action from a sequence.
        
        Args:
            sequence: Input sequence
            threshold: Confidence threshold for prediction
            
        Returns:
            Predicted action label or None if below threshold
        """
        probabilities = self.predict(sequence)
        max_prob = np.max(probabilities)
        
        if max_prob >= threshold:
            predicted_index = np.argmax(probabilities)
            return self.actions[predicted_index]
        
        return None
    
    def get_model_summary(self):
        """Print model summary."""
        if self.model is None:
            logger.warning("Model not built yet")
            return
        
        self.model.summary()


# Default configuration based on notebook
DEFAULT_ACTIONS = np.array(['hello', 'thanks', 'iloveyou'])
SEQUENCE_LENGTH = 30
KEYPOINT_FEATURES = 1662