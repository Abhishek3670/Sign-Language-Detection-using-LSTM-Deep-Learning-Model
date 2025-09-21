#!/usr/bin/env python3
"""
Advanced GPU detection test for CUDA 12.1 + TensorFlow 2.15.0
"""

import os
import sys

def setup_cuda_path():
    """Add CUDA DLL directory to Python path"""
    cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin'
    if os.path.exists(cuda_path):
        try:
            os.add_dll_directory(cuda_path)
            print(f"✅ Added CUDA path: {cuda_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to add CUDA path: {e}")
            return False
    else:
        print(f"❌ CUDA path not found: {cuda_path}")
        return False

def test_tensorflow_detailed():
    """Detailed TensorFlow GPU test"""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check if CUDA is built in
        try:
            cuda_build = tf.test.is_built_with_cuda()
            print(f"Built with CUDA: {cuda_build}")
        except AttributeError:
            print("Built with CUDA: Cannot determine")
        
        # Check GPU availability
        try:
            gpu_available = tf.test.is_gpu_available()
            print(f"GPU available (test): {gpu_available}")
        except AttributeError:
            print("GPU available (test): Method not available")
        
        # List physical devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Physical GPU devices: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    print(f"    Details: {details}")
                except Exception as e:
                    print(f"    Details: {e}")
        
        # Try a simple GPU computation
        if gpus:
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                    print(f"✅ GPU computation successful: {c.numpy()}")
                    return True
            except Exception as e:
                print(f"❌ GPU computation failed: {e}")
        
        return len(gpus) > 0
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def main():
    print("🔍 Advanced GPU Detection Test for CUDA 12.1")
    print("=" * 60)
    
    print("\n1. Setting up CUDA paths...")
    cuda_ok = setup_cuda_path()
    
    print("\n2. Testing TensorFlow GPU support...")
    tf_ok = test_tensorflow_detailed()
    
    print("\n" + "=" * 60)
    print("📋 Final Results:")
    print(f"   CUDA Path Setup: {'✅' if cuda_ok else '❌'}")
    print(f"   TensorFlow GPU: {'✅' if tf_ok else '❌'}")
    
    if not tf_ok:
        print("\n💡 Possible Solutions:")
        print("   1. TensorFlow 2.15.0 Windows wheels may not include CUDA 12.1")
        print("   2. Try installing CUDA 11.8 alongside 12.1")
        print("   3. Use Linux/WSL2 for better TensorFlow GPU support")
        print("   4. Consider building TensorFlow from source")
        print("   5. Wait for official CUDA 12.1 support in future releases")

if __name__ == "__main__":
    main()