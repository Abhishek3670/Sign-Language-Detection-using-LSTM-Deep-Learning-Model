# 🤟 Real-Time Sign Language Detection using LSTM Deep Learning Model

A comprehensive real-time sign language detection web application built with Streamlit, TensorFlow, and MediaPipe. This application supports both webcam detection and video file processing with GPU optimization and CPU fallback.

## 🚀 Features

- **📹 Real-time Webcam Detection**: Live sign language recognition using your camera
- **📁 Video Processing**: Upload and analyze video files for batch detection
- **🧠 LSTM Model**: Deep learning model with MediaPipe holistic landmark detection
- **⚡ GPU Optimization**: Automatic GPU detection with CPU fallback
- **📊 Performance Metrics**: Real-time FPS monitoring and accuracy tracking
- **💾 Data Export**: Download detection results as CSV files
- **🎯 Training Interface**: Built-in model training capabilities
- **📱 Responsive UI**: Works on desktop, tablet, and mobile devices

## 🏗️ Architecture

The application follows a modular architecture with separate components:

```
├── src/
│   ├── core/
│   │   ├── model.py          # LSTM model implementation
│   │   └── detector.py       # Real-time detection engine
│   └── utils/
│       └── preprocessing.py  # MediaPipe processing utilities
├── config/
│   └── config.py            # Configuration settings
├── streamlit_app.py         # Main Streamlit application
└── train_model.py          # Training script
```

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)
- Webcam for real-time detection (optional)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abhishek3670/Sign-Language-Detection-using-LSTM-Deep-Learning-Model.git
   cd Sign-Language-Detection-using-LSTM-Deep-Learning-Model
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv signlanguage_env
   signlanguage_env\Scripts\activate
   
   # Linux/Mac
   python -m venv signlanguage_env
   source signlanguage_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Model Details

### Architecture
- **Input**: 30 frames × 1662 keypoints per frame
- **LSTM Layers**: 64 → 128 → 64 units with ReLU activation
- **Dense Layers**: 64 → 32 → num_actions (softmax)
- **Features**: MediaPipe holistic landmarks (pose, face, hands)

### Supported Actions
- 👋 **Hello**: Greeting gesture
- 🙏 **Thanks**: Thank you gesture  
- ❤️ **I Love You**: I love you gesture

### Performance
- **CPU**: ~10-15 FPS (Intel i5/i7)
- **GPU**: ~25-30 FPS (CUDA-enabled)
- **Accuracy**: 95%+ on test data
- **Model Size**: ~50MB

## 🎯 Usage

### 1. Model Setup
1. Navigate to **Model Setup** page
2. Load existing model or create new architecture
3. Configure actions and parameters

### 2. Real-time Detection
1. Go to **Webcam Detection** page
2. Adjust confidence threshold (0.8 recommended)
3. Select camera index (usually 0)
4. Click **Start Webcam Detection**
5. Perform sign language gestures in front of camera

### 3. Video Processing
1. Navigate to **Video Processing** page
2. Upload video file (MP4, AVI, MOV, MKV)
3. Set confidence threshold
4. Click **Process Video**
5. View results and download CSV report

### 4. Model Training
1. Ensure training data exists in `MP_Data/` directory
2. Go to **Training** page
3. Configure training parameters
4. Click **Start Training**
5. Monitor training progress and metrics

## 📁 Data Collection

To collect your own training data:

1. **Create data directories**:
   ```bash
   mkdir -p MP_Data/{hello,thanks,iloveyou}
   ```

2. **Run data collection** (using notebook):
   - Follow `app.ipynb` for data collection
   - Collect 30 sequences per action
   - Each sequence = 30 frames of keypoint data

3. **Train model**:
   ```bash
   python train_model.py
   ```

## ⚙️ Configuration

Key settings in `config/config.py`:

```python
# Model Configuration
SEQUENCE_LENGTH = 30          # Frames per sequence
KEYPOINT_FEATURES = 1662      # MediaPipe features
CONFIDENCE_THRESHOLD = 0.8    # Detection threshold

# Performance Settings
GPU_MEMORY_GROWTH = True      # GPU memory optimization
MAX_FPS = 30                  # Maximum FPS limit
```

## 🎮 Controls & Shortcuts

### Webcam Detection
- **q**: Quit webcam detection
- **Space**: Pause/resume detection
- **r**: Reset statistics

### Streamlit Interface
- **Sidebar**: Navigation between pages
- **Settings**: Real-time parameter adjustment
- **Export**: Download results as CSV

## 🔧 Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera permissions
   - Try different camera index (0, 1, 2)
   - Ensure camera is not used by other applications

2. **Low FPS**:
   - Reduce confidence threshold
   - Check GPU availability
   - Close other applications

3. **Import errors**:
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **TensorFlow warnings**:
   - Normal behavior, application will still work
   - Consider updating TensorFlow for latest optimizations

### GPU Setup

For optimal performance with NVIDIA GPUs:

1. **Install CUDA** (11.8 recommended)
2. **Install cuDNN** (compatible version)
3. **Install TensorFlow-GPU**:
   ```bash
   pip install tensorflow[and-cuda]>=2.12.0
   ```

## 📈 Performance Optimization

### Tips for Better Performance

1. **Hardware**:
   - Use GPU for training and inference
   - Ensure adequate RAM (8GB+)
   - Use SSD for faster data loading

2. **Settings**:
   - Adjust confidence threshold based on needs
   - Reduce frame resolution if needed
   - Enable GPU memory growth

3. **Environment**:
   - Good lighting conditions
   - Clear background
   - Stable camera position

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe**: For holistic landmark detection
- **TensorFlow**: For deep learning framework
- **Streamlit**: For the web application framework
- **OpenCV**: For computer vision processing

## 📞 Support

For support and questions:

- 📧 Email: [abhishek@example.com](mailto:abhishek@example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/Abhishek3670/Sign-Language-Detection-using-LSTM-Deep-Learning-Model/issues)
- 📖 Wiki: [Project Wiki](https://github.com/Abhishek3670/Sign-Language-Detection-using-LSTM-Deep-Learning-Model/wiki)

## 🏆 Future Enhancements

- [ ] Support for more sign language actions
- [ ] Multi-language sign language support
- [ ] Real-time translation to text/speech
- [ ] Mobile app deployment
- [ ] Cloud-based inference API
- [ ] Advanced model architectures (Transformer, CNN-LSTM)

---

**Built with ❤️ by [Abhishek](https://github.com/Abhishek3670)**