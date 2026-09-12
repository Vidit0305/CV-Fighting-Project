"""
Hand Detector Module for CV Fighter
Uses Google MediaPipe Hands for high-accuracy, real-time 21-landmark detection.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import mediapipe as mp


@dataclass
class HandData:
    """Represents complete detection data for a single hand in a frame."""
    # 21 normalized landmarks (x: [0,1], y: [0,1], z: depth)
    landmarks_norm: List[Tuple[float, float, float]] = field(default_factory=list)
    # 21 pixel landmarks (x_px, y_px)
    landmarks_px: List[Tuple[int, int]] = field(default_factory=list)
    # MediaPipe classified handedness ("Left" or "Right")
    mp_handedness: str = "Unknown"
    # MediaPipe classification confidence
    mp_confidence: float = 0.0
    # Screen side assignment based on horizontal position ("left" or "right")
    screen_side: str = "left"
    # Bounding box in pixel coordinates: (min_x, min_y, max_x, max_y)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    # Normalized palm center (centroid of wrist + MCPs)
    palm_center_norm: Tuple[float, float] = (0.0, 0.0)
    # Pixel palm center (x, y)
    palm_center_px: Tuple[int, int] = (0, 0)
    # Normalized wrist position (landmark 0)
    wrist_norm: Tuple[float, float] = (0.0, 0.0)
    # Pixel wrist position
    wrist_px: Tuple[int, int] = (0, 0)


class HandDetector:
    """
    Wraps MediaPipe Hands to process video frames and extract structured
    information for both hands simultaneously.
    """

    # Landmark connections for rendering
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (9, 10), (10, 11), (11, 12),
        # Ring finger
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17)
    ]

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.60,
    ):
        """
        Initialize the MediaPipe Hands detector.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray, mirror: bool = True) -> Tuple[np.ndarray, List[HandData]]:
        """
        Processes a single BGR camera frame and returns:
        - The processed frame (mirrored if mirror=True)
        - A list of HandData objects for all detected hands
        """
        if mirror:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        # MediaPipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        detected_hands: List[HandData] = []

        if results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                # Retrieve handedness classification if provided by MediaPipe
                mp_handedness = "Unknown"
                mp_confidence = 0.0
                if results.multi_handedness and idx < len(results.multi_handedness):
                    classification = results.multi_handedness[idx].classification[0]
                    # Note: MediaPipe assumes an unmirrored camera input.
                    # In a mirrored selfie feed, MediaPipe's "Left" hand corresponds to the user's right hand.
                    # We invert the label when mirror=True to match physical user perspective.
                    raw_label = classification.label
                    if mirror:
                        mp_handedness = "Right" if raw_label == "Left" else "Left"
                    else:
                        mp_handedness = raw_label
                    mp_confidence = float(classification.score)

                # Extract 21 landmark points
                lms_norm: List[Tuple[float, float, float]] = []
                lms_px: List[Tuple[int, int]] = []
                xs_px: List[int] = []
                ys_px: List[int] = []

                for lm in hand_lms.landmark:
                    lms_norm.append((lm.x, lm.y, lm.z))
                    px_x = int(np.clip(lm.x * w, 0, w - 1))
                    px_y = int(np.clip(lm.y * h, 0, h - 1))
                    lms_px.append((px_x, px_y))
                    xs_px.append(px_x)
                    ys_px.append(px_y)

                # Bounding box
                min_x = max(0, min(xs_px) - 10)
                max_x = min(w - 1, max(xs_px) + 10)
                min_y = max(0, min(ys_px) - 10)
                max_y = min(h - 1, max(ys_px) + 10)
                bbox = (min_x, min_y, max_x, max_y)

                # Calculate palm center using wrist (0) and MCP joints (1, 5, 9, 13, 17)
                palm_indices = [0, 1, 5, 9, 13, 17]
                avg_norm_x = sum(lms_norm[i][0] for i in palm_indices) / len(palm_indices)
                avg_norm_y = sum(lms_norm[i][1] for i in palm_indices) / len(palm_indices)
                palm_center_norm = (avg_norm_x, avg_norm_y)
                palm_center_px = (int(avg_norm_x * w), int(avg_norm_y * h))

                wrist_norm = (lms_norm[0][0], lms_norm[0][1])
                wrist_px = lms_px[0]

                # Determine screen side (left half vs right half of the image)
                screen_side = "left" if palm_center_norm[0] < 0.5 else "right"

                hand_data = HandData(
                    landmarks_norm=lms_norm,
                    landmarks_px=lms_px,
                    mp_handedness=mp_handedness,
                    mp_confidence=mp_confidence,
                    screen_side=screen_side,
                    bbox=bbox,
                    palm_center_norm=palm_center_norm,
                    palm_center_px=palm_center_px,
                    wrist_norm=wrist_norm,
                    wrist_px=wrist_px,
                )
                detected_hands.append(hand_data)

        return frame, detected_hands

    def draw_skeleton(
        self,
        frame: np.ndarray,
        hand: HandData,
        bone_color: Tuple[int, int, int] = (0, 200, 255),
        joint_color: Tuple[int, int, int] = (255, 255, 255),
        palm_color: Tuple[int, int, int] = (0, 255, 100),
        draw_labels: bool = False,
    ) -> None:
        """
        Renders sleek skeleton bones, joints, and palm center for a detected hand.
        """
        # 1. Draw connecting bones
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            pt1 = hand.landmarks_px[start_idx]
            pt2 = hand.landmarks_px[end_idx]
            cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        # 2. Draw joints
        for idx, (px, py) in enumerate(hand.landmarks_px):
            # Key tips: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
            is_tip = idx in (4, 8, 12, 16, 20)
            radius = 5 if is_tip else 3
            cv2.circle(frame, (px, py), radius, joint_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

            if draw_labels:
                cv2.putText(
                    frame,
                    str(idx),
                    (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Draw palm centroid
        cx, cy = hand.palm_center_px
        cv2.circle(frame, (cx, cy), 6, palm_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
