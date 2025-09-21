"""
Configuration settings for the sign language detection application.
"""

import numpy as np
import os
import logging
from typing import Literal, Dict, Any

# Model Configuration
SEQUENCE_LENGTH = 30  # Number of frames in each sequence
KEYPOINT_FEATURES = 1662  # Total keypoint features per frame
DEFAULT_ACTIONS = np.array(['hello', 'thanks', 'iloveyou'])  # Supported actions

# File Paths
MODEL_PATH = 'action.h5'
DATA_PATH = 'MP_Data'
LOGS_PATH = 'Logs'

# Detection Settings
CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence for predictions
MIN_DETECTION_CONFIDENCE = 0.5  # MediaPipe detection confidence
MIN_TRACKING_CONFIDENCE = 0.5   # MediaPipe tracking confidence

# Training Configuration
DEFAULT_EPOCHS = 100
TRAIN_TEST_SPLIT = 0.05  # 5% for testing
RANDOM_STATE = 42

# Real-time Processing
WEBCAM_INDEX = 0
MAX_FPS = 30
FRAME_BUFFER_SIZE = 10  # For prediction smoothing

# Streamlit Configuration with explicit types
STREAMLIT_PAGE_TITLE: str = 'Sign Language Detection'
STREAMLIT_PAGE_ICON: str = '🤟'
STREAMLIT_LAYOUT: Literal['wide', 'centered'] = 'wide'
STREAMLIT_SIDEBAR_STATE: Literal['auto', 'expanded', 'collapsed'] = 'expanded'

# Legacy config dict (deprecated - use individual constants instead)
STREAMLIT_CONFIG = {
    'page_title': STREAMLIT_PAGE_TITLE,
    'page_icon': STREAMLIT_PAGE_ICON,
    'layout': STREAMLIT_LAYOUT,
    'initial_sidebar_state': STREAMLIT_SIDEBAR_STATE
}

# UI Colors (BGR format for OpenCV)
COLORS = {
    'primary': (245, 117, 66),    # Orange
    'secondary': (117, 245, 16),  # Green  
    'tertiary': (16, 117, 245),   # Blue
    'text': (255, 255, 255),      # White
    'success': (0, 255, 0),       # Green
    'warning': (0, 255, 255),     # Yellow
    'error': (0, 0, 255)          # Red
}

# Performance Settings
GPU_MEMORY_GROWTH = True
ENABLE_MIXED_PRECISION = False  # Can be enabled for RTX cards

# Logging Configuration
LOGGING_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}


def get_model_config():
    """Get model configuration dictionary."""
    return {
        'sequence_length': SEQUENCE_LENGTH,
        'keypoint_features': KEYPOINT_FEATURES,
        'actions': DEFAULT_ACTIONS,
        'model_path': MODEL_PATH
    }


def get_detection_config():
    """Get detection configuration dictionary."""
    return {
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'min_detection_confidence': MIN_DETECTION_CONFIDENCE,
        'min_tracking_confidence': MIN_TRACKING_CONFIDENCE,
        'webcam_index': WEBCAM_INDEX,
        'colors': list(COLORS.values())[:len(DEFAULT_ACTIONS)]
    }


def get_training_config():
    """Get training configuration dictionary."""
    return {
        'epochs': DEFAULT_EPOCHS,
        'test_size': TRAIN_TEST_SPLIT,
        'random_state': RANDOM_STATE,
        'data_path': DATA_PATH,
        'logs_path': LOGS_PATH
    }


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [DATA_PATH, LOGS_PATH]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Action-specific configurations
ACTION_CONFIGS = {
    'hello': {
        'color': COLORS['primary'],
        'description': 'Hello greeting gesture'
    },
    'thanks': {
        'color': COLORS['secondary'], 
        'description': 'Thank you gesture'
    },
    'iloveyou': {
        'color': COLORS['tertiary'],
        'description': 'I love you gesture'
    }
}

# Development flags
DEBUG_MODE = False
SAVE_PROCESSED_FRAMES = False
ENABLE_PROFILING = False