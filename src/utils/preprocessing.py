"""
Preprocessing utilities for sign language detection.
Extracted from notebook to provide modular, reusable functions.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple


class MediaPipeProcessor:
    """MediaPipe holistic processing for landmark detection."""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def mediapipe_detection(self, image: np.ndarray) -> Tuple[np.ndarray, object]:
        """
        Perform MediaPipe detection on input image.
        
        Args:
            image: Input BGR image from OpenCV
            
        Returns:
            Tuple of (processed_image, results)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Make detection
        results = self.holistic.process(image_rgb)
        
        # Convert back to BGR
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        return image_bgr, results
    
    def draw_styled_landmarks(self, image: np.ndarray, results) -> np.ndarray:
        """
        Draw styled landmarks on the image.
        
        Args:
            image: Input image
            results: MediaPipe results
            
        Returns:
            Image with drawn landmarks
        """
        # Draw face connections
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
            self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
        
        # Draw pose connections
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
        
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
        
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        
        return image
    
    def extract_keypoints(self, results) -> np.ndarray:
        """
        Extract keypoints from MediaPipe results.
        
        Args:
            results: MediaPipe detection results
            
        Returns:
            Flattened numpy array of keypoints (1662 features total)
        """
        # Extract pose landmarks (33 landmarks * 4 features = 132)
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33 * 4)
        
        # Extract face landmarks (468 landmarks * 3 features = 1404)
        face = np.array([[res.x, res.y, res.z] 
                        for res in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468 * 3)
        
        # Extract left hand landmarks (21 landmarks * 3 features = 63)
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21 * 3)
        
        # Extract right hand landmarks (21 landmarks * 3 features = 63)
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21 * 3)
        
        # Concatenate all features: 132 + 1404 + 63 + 63 = 1662
        return np.concatenate([pose, face, lh, rh])
    
    def close(self):
        """Close MediaPipe resources."""
        if hasattr(self, 'holistic'):
            self.holistic.close()


def prob_viz(res: np.ndarray, actions: np.ndarray, input_frame: np.ndarray, 
            colors: list) -> np.ndarray:
    """
    Visualize prediction probabilities on the frame.
    
    Args:
        res: Prediction probabilities
        actions: Action labels
        input_frame: Input image frame
        colors: List of colors for each action
        
    Returns:
        Frame with probability visualization
    """
    output_frame = input_frame.copy()
    
    for num, prob in enumerate(res):
        if num < len(colors):  # Ensure we don't exceed color list bounds
            # Draw probability bar
            cv2.rectangle(output_frame, (0, 60 + num * 40), 
                         (int(prob * 100), 90 + num * 40), colors[num], -1)
            
            # Draw action label
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return output_frame