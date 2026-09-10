import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

def extract_landmarks_from_video(video_path: str, target_frames: int = 32) -> np.ndarray:
    """
    Processes an MP4 video clip, extracts Left Hand (21), Right Hand (21),
    and Pose (33) landmarks using MediaPipe Holistic, and resamples the 
    resulting sequence to target_frames (shape: [target_frames, 225]).
    """
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            # Left Hand (21 landmarks x 3 = 63)
            if results.left_hand_landmarks:
                lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
            else:
                lh = np.zeros(63, dtype=np.float32)

            # Right Hand (21 landmarks x 3 = 63)
            if results.right_hand_landmarks:
                rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
            else:
                rh = np.zeros(63, dtype=np.float32)

            # Pose (33 landmarks x 3 = 99)
            if results.pose_landmarks:
                pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
            else:
                pose = np.zeros(99, dtype=np.float32)

            # Concatenate to form a 225-element vector for this frame
            frame_features = np.concatenate([lh, rh, pose])
            frames_landmarks.append(frame_features)

    cap.release()

    # Handle empty or unreadable video
    if len(frames_landmarks) == 0:
        return np.zeros((target_frames, 225), dtype=np.float32)

    seq = np.array(frames_landmarks, dtype=np.float32)
    n_frames = len(seq)

    # Resample or pad sequence to target_frames (32)
    if n_frames == target_frames:
        return seq
    elif n_frames > target_frames:
        idx = np.linspace(0, n_frames - 1, target_frames, dtype=int)
        return seq[idx]
    else:
        pad_size = target_frames - n_frames
        return np.pad(seq, ((0, pad_size), (0, 0)), mode="constant")