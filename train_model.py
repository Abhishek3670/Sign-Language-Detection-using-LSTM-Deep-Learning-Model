"""
Training script for the sign language detection model.
Uses existing MP_Data to train the LSTM model.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.model import SignLanguageLSTM
from config.config import *

def main():
    """Main training function."""
    print("🚀 Starting Sign Language Model Training")
    print("=" * 50)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Available: {len(gpus) > 0}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
    else:
        print("Using CPU for training")
    
    # Configure GPU if available
    if gpus and GPU_MEMORY_GROWTH:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth configured")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration warning: {e}")
    
    # Create model
    print(f"\n📊 Model Configuration:")
    print(f"Actions: {DEFAULT_ACTIONS}")
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print(f"Keypoint Features: {KEYPOINT_FEATURES}")
    
    model = SignLanguageLSTM(DEFAULT_ACTIONS)
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Training data not found at {DATA_PATH}")
        print("Please run data collection first or ensure MP_Data directory exists")
        return
    
    # Check available actions in data
    available_actions = []
    for action in DEFAULT_ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if os.path.exists(action_path):
            sequences = len([d for d in os.listdir(action_path) 
                           if os.path.isdir(os.path.join(action_path, d))])
            if sequences > 0:
                available_actions.append(action)
                print(f"✅ {action}: {sequences} sequences")
            else:
                print(f"⚠️ {action}: No sequences found")
        else:
            print(f"❌ {action}: Directory not found")
    
    if not available_actions:
        print("❌ No training data found for any actions")
        return
    
    # Update actions to only include available ones
    if len(available_actions) < len(DEFAULT_ACTIONS):
        print(f"\n⚠️ Using only available actions: {available_actions}")
        model = SignLanguageLSTM(np.array(available_actions))
    
    try:
        # Prepare data
        print(f"\n📚 Preparing training data from {DATA_PATH}...")
        X_train, X_test, y_train, y_test = model.prepare_data(DATA_PATH)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        # Build and train model
        print(f"\n🏗️ Building model architecture...")
        model.build_model()
        model.get_model_summary()
        
        print(f"\n🚀 Starting training for {DEFAULT_EPOCHS} epochs...")
        print("This may take a while depending on your hardware...")
        
        # Train with reduced epochs for quick testing
        history = model.train(X_train, y_train, epochs=min(50, DEFAULT_EPOCHS), verbose=1)
        
        # Evaluate on test data
        if len(X_test) > 0 and model.model is not None:
            print(f"\n📊 Evaluating on test data...")
            test_loss, test_accuracy = model.model.evaluate(X_test, y_test, verbose=0)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
        else:
            print("\n⚠️ Skipping evaluation (no test data or model not ready)")
        
        # Save model
        print(f"\n💾 Saving model to {MODEL_PATH}...")
        model.save_model(MODEL_PATH)
        
        print("\n✅ Training completed successfully!")
        print(f"Model saved as: {MODEL_PATH}")
        print("You can now use the model in the Streamlit app")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()