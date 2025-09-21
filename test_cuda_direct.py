#!/usr/bin/env python3
"""
Direct CUDA and TensorFlow GPU test script.
Tests various methods to detect GPU availability.
"""

import subprocess
import sys

def test_nvidia_smi():
    """Test nvidia-smi command"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi working")
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except Exception as e:
        print(f"❌ nvidia-smi error: {e}")
        return False

def test_nvcc():
    """Test NVCC compiler"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVCC available")
            print(f"   {result.stdout.split('release')[1].split(',')[0].strip()}")
            return True
        else:
            print("❌ NVCC not found")
            return False
    except Exception as e:
        print(f"❌ NVCC error: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow GPU detection"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported")
        
        # Check build configuration
        build_info = tf.sysconfig.get_build_info()
        print(f"   CUDA build: {build_info.get('is_cuda_build', 'Unknown')}")
        print(f"   CUDA version: {build_info.get('cuda_version', 'Unknown')}")
        print(f"   cuDNN version: {build_info.get('cudnn_version', 'Unknown')}")
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   GPU devices: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"     GPU {i}: {gpu.name}")
            return True
        else:
            print("   No GPU devices detected")
            return False
            
    except Exception as e:
        print(f"❌ TensorFlow error: {e}")
        return False

def main():
    print("🔍 GPU Detection Test")
    print("=" * 50)
    
    print("\n1. NVIDIA Driver Test:")
    nvidia_ok = test_nvidia_smi()
    
    print("\n2. CUDA Compiler Test:")
    nvcc_ok = test_nvcc()
    
    print("\n3. TensorFlow GPU Test:")
    tf_ok = test_tensorflow()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print(f"   NVIDIA Driver: {'✅' if nvidia_ok else '❌'}")
    print(f"   CUDA Compiler: {'✅' if nvcc_ok else '❌'}")
    print(f"   TensorFlow GPU: {'✅' if tf_ok else '❌'}")
    
    if nvidia_ok and nvcc_ok and not tf_ok:
        print("\n💡 Recommendation:")
        print("   GPU hardware and CUDA are working, but TensorFlow")
        print("   doesn't have GPU support. This suggests a CPU-only")
        print("   TensorFlow build was installed.")
        print("   ")
        print("   Try installing: pip install tensorflow[and-cuda]")

if __name__ == "__main__":
    main()