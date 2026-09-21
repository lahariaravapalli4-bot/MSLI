import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
NUM_FEATURES = 225  # 21 Left Hand (63) + 21 Right Hand (63) + 33 Pose (99)


def normalize_frame_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Applies torso-relative spatial translation and distance scaling.
    Matches the normalization logic in train.py exactly.
    
    coords: 1D array of 225 floats representing (x, y, z):
      - indices 0..62:   Left Hand (21 keypoints * 3)
      - indices 63..125:  Right Hand (21 keypoints * 3)
      - indices 126..224: Upper Pose (33 keypoints * 3)
    """
    # Keypoint 11 (Left Shoulder) in pose slice: offset = 126 + (11 * 3) = 159
    ls_x = coords[159]
    ls_y = coords[160]

    # Keypoint 12 (Right Shoulder) in pose slice: offset = 126 + (12 * 3) = 162
    rs_x = coords[162]
    rs_y = coords[163]

    # If shoulders are detected, anchor to shoulder center
    if (ls_x != 0.0 or ls_y != 0.0) and (rs_x != 0.0 or rs_y != 0.0):
        anchor_x = (ls_x + rs_x) / 2.0
        anchor_y = (ls_y + rs_y) / 2.0
        torso_scale = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)
        if torso_scale < 1e-4:
            torso_scale = 1.0
    else:
        # Fallback if pose tracking dropped shoulders in that specific frame
        anchor_x = 0.5
        anchor_y = 0.5
        torso_scale = 1.0

    normalized = coords.copy()

    # Shift (x, y) relative to torso anchor and scale by body size
    for i in range(0, NUM_FEATURES, 3):
        # Keep empty/missing joints at exact 0.0
        if coords[i] != 0.0 or coords[i + 1] != 0.0:
            normalized[i] = (coords[i] - anchor_x) / torso_scale
            normalized[i + 1] = (coords[i + 1] - anchor_y) / torso_scale
            # z coordinate remains as relative depth

    return normalized


def extract_landmarks_from_video(video_path: str, target_frames: int = 32) -> np.ndarray:
    """
    Processes an MP4 video clip, extracts Left Hand (21), Right Hand (21),
    and Pose (33) landmarks using MediaPipe Holistic, applies torso-anchor
    normalization, and standardizes the sequence to target_frames (shape: [target_frames, 225]).
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

            # Concatenate to form raw 225-element vector
            raw_frame_features = np.concatenate([lh, rh, pose])

            # Apply identical torso normalization
            norm_frame_features = normalize_frame_coordinates(raw_frame_features)
            frames_landmarks.append(norm_frame_features)

    cap.release()

    # Handle empty or unreadable video
    if len(frames_landmarks) == 0:
        return np.zeros((target_frames, NUM_FEATURES), dtype=np.float32)

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