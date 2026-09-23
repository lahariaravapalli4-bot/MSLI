import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
NUM_FEATURES = 225  # 21 Left Hand (63) + 33 Pose (99) + 21 Right Hand (63)


def normalize_frame_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Applies torso-relative spatial translation and distance scaling
    matching the exact implementation used in model/train.py.
    """
    # Keypoint 11 (Left Shoulder) in pose slice: offset = 126 + (11 * 3) = 159
    ls_x = coords[159]
    ls_y = coords[160]

    # Keypoint 12 (Right Shoulder) in pose slice: offset = 126 + (12 * 3) = 162
    rs_x = coords[162]
    rs_y = coords[163]

    # If anchor coordinates are valid, compute center and Euclidean distance
    if (ls_x != 0.0 or ls_y != 0.0) and (rs_x != 0.0 or rs_y != 0.0):
        anchor_x = (ls_x + rs_x) / 2.0
        anchor_y = (ls_y + rs_y) / 2.0
        torso_scale = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)
        if torso_scale < 1e-4:
            torso_scale = 1.0
    else:
        anchor_x = 0.5
        anchor_y = 0.5
        torso_scale = 1.0

    normalized = coords.copy()

    # Shift (x, y) relative to torso anchor and scale by body size
    for i in range(0, NUM_FEATURES, 3):
        if coords[i] != 0.0 or coords[i + 1] != 0.0:
            normalized[i] = (coords[i] - anchor_x) / torso_scale
            normalized[i + 1] = (coords[i + 1] - anchor_y) / torso_scale
            # z coordinate remains relative depth

    return normalized


def extract_landmarks_from_video(video_path: str, target_frames: int = 32) -> np.ndarray:
    """
    Extracts Left Hand, Pose, and Right Hand landmark sequences from a video,
    normalizes coordinates, handles dropped detections via forward-fill, and
    samples or pads the sequence to exactly 32 frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []

    # Persistence buffers to prevent dropped frames from zero-snapping
    last_lh = np.zeros(63, dtype=np.float32)
    last_pose = np.zeros(99, dtype=np.float32)
    last_rh = np.zeros(63, dtype=np.float32)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            # 1. Left Hand: 21 landmarks -> 63 floats
            if results.left_hand_landmarks:
                last_lh = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                    dtype=np.float32
                ).flatten()
            lh = last_lh.copy()

            # 2. Pose: 33 landmarks -> 99 floats
            if results.pose_landmarks:
                last_pose = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                    dtype=np.float32
                ).flatten()
            pose = last_pose.copy()

            # 3. Right Hand: 21 landmarks -> 63 floats
            if results.right_hand_landmarks:
                last_rh = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                    dtype=np.float32
                ).flatten()
            rh = last_rh.copy()

            # Concatenate matching dataset ordering: left_hand (63) + pose (99) + right_hand (63)
            raw_frame = np.concatenate([lh, pose, rh])
            norm_frame = normalize_frame_coordinates(raw_frame)
            frames_landmarks.append(norm_frame)

    cap.release()

    if len(frames_landmarks) == 0:
        return np.zeros((target_frames, NUM_FEATURES), dtype=np.float32)
    
    seq = np.array(frames_landmarks, dtype=np.float32)

    # Backfill: If the gesture started with zeros before first detection, fill backwards
    for col_slice in [slice(0, 63), slice(63, 162), slice(162, 225)]:
        sub = seq[:, col_slice]
        # Find first non-zero frame
        non_zero_indices = np.where(np.any(sub != 0.0, axis=1))[0]
        if len(non_zero_indices) > 0:
            first_valid_idx = non_zero_indices[0]
            if first_valid_idx > 0:
                first_valid_frame = sub[first_valid_idx].copy()
                for i in range(first_valid_idx):
                    seq[i, col_slice] = first_valid_frame

    seq = np.array(frames_landmarks, dtype=np.float32)
    n_frames = len(seq)

    # Standardize to fixed 32-frame buffer
    if n_frames == target_frames:
        return seq
    elif n_frames > target_frames:
        idx = np.linspace(0, n_frames - 1, target_frames, dtype=int)
        return seq[idx]
    else:
        pad_size = target_frames - n_frames
        return np.pad(seq, ((0, pad_size), (0, 0)), mode="constant")