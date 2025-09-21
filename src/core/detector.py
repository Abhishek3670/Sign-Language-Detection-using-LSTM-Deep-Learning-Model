"""
Real-time sign language detection using webcam or video files.
"""

import cv2
import numpy as np
import time
from typing import Optional, Callable, Tuple
from collections import deque
import logging

from ..utils.preprocessing import MediaPipeProcessor, prob_viz
from ..core.model import SignLanguageLSTM

logger = logging.getLogger(__name__)


class RealTimeDetector:
    """Real-time sign language detection system."""
    
    def __init__(self, model: SignLanguageLSTM, confidence_threshold: float = 0.8):
        """
        Initialize the real-time detector.
        
        Args:
            model: Trained SignLanguageLSTM model
            confidence_threshold: Minimum confidence for predictions
        """
        self.model = model
        self.processor = MediaPipeProcessor()
        self.confidence_threshold = confidence_threshold
        
        # Sequence buffer for collecting frames
        self.sequence_buffer = deque(maxlen=model.sequence_length)
        self.prediction_history = deque(maxlen=10)  # For smoothing predictions
        
        # Colors for visualization (matching notebook)
        self.colors = [(245, 117, 66), (117, 245, 16), (16, 117, 245)]
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str], np.ndarray]:
        """
        Process a single frame for sign language detection.
        
        Args:
            frame: Input frame from camera/video
            
        Returns:
            Tuple of (processed_frame, predicted_action, probabilities)
        """
        # Perform MediaPipe detection
        processed_frame, results = self.processor.mediapipe_detection(frame)
        
        # Draw landmarks
        processed_frame = self.processor.draw_styled_landmarks(processed_frame, results)
        
        # Extract keypoints
        keypoints = self.processor.extract_keypoints(results)
        self.sequence_buffer.append(keypoints)
        
        predicted_action = None
        probabilities = np.zeros(len(self.model.actions))
        
        # Make prediction if we have enough frames
        if len(self.sequence_buffer) == self.model.sequence_length:
            sequence = np.array(list(self.sequence_buffer))
            
            try:
                probabilities = self.model.predict(sequence)
                max_prob = np.max(probabilities)
                
                if max_prob >= self.confidence_threshold:
                    predicted_index = np.argmax(probabilities)
                    predicted_action = self.model.actions[predicted_index]
                    
                    # Add to prediction history for smoothing
                    self.prediction_history.append(predicted_action)
                    
                    # Use most common recent prediction for stability
                    if len(self.prediction_history) >= 3:
                        predicted_action = max(set(self.prediction_history), 
                                             key=list(self.prediction_history).count)
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
        
        # Visualize probabilities
        if len(probabilities) > 0:
            processed_frame = prob_viz(probabilities, self.model.actions, 
                                     processed_frame, self.colors[:len(self.model.actions)])
        
        # Add prediction text
        if predicted_action:
            cv2.putText(processed_frame, f"Action: {predicted_action}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add confidence text
        if len(probabilities) > 0:
            max_confidence = np.max(probabilities)
            cv2.putText(processed_frame, f"Confidence: {max_confidence:.2f}", 
                       (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        return processed_frame, predicted_action, probabilities
    
    def run_webcam_detection(self, camera_index: int = 0, 
                           frame_callback: Optional[Callable] = None) -> None:
        """
        Run real-time detection using webcam.
        
        Args:
            camera_index: Camera index (usually 0 for default camera)
            frame_callback: Optional callback function for each processed frame
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        logger.info("Starting webcam detection. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame, predicted_action, probabilities = self.process_frame(frame)
                
                # Calculate and display FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                           (processed_frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Call frame callback if provided
                if frame_callback:
                    frame_callback(processed_frame, predicted_action, probabilities)
                
                # Display frame
                cv2.imshow('Sign Language Detection', processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.processor.close()
            logger.info("Webcam detection stopped")
    
    def process_video_file(self, video_path: str, 
                          frame_callback: Optional[Callable] = None) -> list:
        """
        Process a video file for sign language detection.
        
        Args:
            video_path: Path to the video file
            frame_callback: Optional callback for each processed frame
            
        Returns:
            List of predictions for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        predictions = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path} ({total_frames} frames)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, predicted_action, probabilities = self.process_frame(frame)
                
                predictions.append({
                    'frame': frame_count,
                    'action': predicted_action,
                    'probabilities': probabilities.tolist() if len(probabilities) > 0 else [],
                    'max_confidence': float(np.max(probabilities)) if len(probabilities) > 0 else 0.0
                })
                
                # Call frame callback if provided
                if frame_callback:
                    frame_callback(processed_frame, predicted_action, probabilities)
                
                frame_count += 1
                
                # Log progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processing progress: {progress:.1f}%")
                    
        finally:
            cap.release()
            self.processor.close()
            logger.info(f"Video processing completed. Processed {frame_count} frames")
        
        return predictions
    
    def get_performance_stats(self) -> dict:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'elapsed_time': elapsed_time,
            'average_fps': fps,
            'sequence_buffer_size': len(self.sequence_buffer),
            'prediction_history_size': len(self.prediction_history)
        }
    
    def reset_buffers(self):
        """Reset sequence and prediction buffers."""
        self.sequence_buffer.clear()
        self.prediction_history.clear()
        self.frame_count = 0
        self.start_time = time.time()
    
    def close(self):
        """Clean up resources."""
        self.processor.close()


def create_detector(model_path: str = 'action.h5', 
                   actions: Optional[np.ndarray] = None) -> RealTimeDetector:
    """
    Create a real-time detector with a loaded model.
    
    Args:
        model_path: Path to the saved model
        actions: Action labels (will use default if None)
        
    Returns:
        Configured RealTimeDetector
    """
    if actions is None:
        # Use default actions from notebook
        actions = np.array(['hello', 'thanks', 'iloveyou'])
    
    # Create and load model
    lstm_model = SignLanguageLSTM(actions)
    lstm_model.load_model(model_path)
    
    # Create detector
    detector = RealTimeDetector(lstm_model)
    
    return detector