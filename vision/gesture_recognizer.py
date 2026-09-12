"""
Gesture Recognizer Module for CV Fighter
Recognizes combat action gestures with geometric invariance and temporal debouncing:
  1. OPEN_PALM   -> Attack 1 (Y)
  2. CLOSED_FIST  -> Attack 2 (H)
  3. V_SIGN       -> Attack 3 (G)
  4. THUMBS_UP    -> Attack 4 (J)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import time
from vision.hand_detector import HandData


@dataclass
class GestureState:
    """Tracks internal state, stability counter, and timestamps for gesture detection."""
    # Active confirmed gesture (debounced)
    confirmed_gesture: str = "NONE"
    # Raw candidate gesture from the current frame
    raw_gesture: str = "NONE"
    # Raw confidence of candidate gesture [0.0 - 1.0]
    confidence: float = 0.0
    # Stability counter (consecutive frames candidate has appeared)
    stability_count: int = 0
    # True if the gesture has achieved required stability frames
    is_stable: bool = False
    # States of the 5 fingers (True = extended, False = curled)
    finger_states: Dict[str, bool] = field(
        default_factory=lambda: {
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
        }
    )


class GestureRecognizer:
    """
    Analyzes 21 hand landmarks to classify combat gestures using scale-invariant
    Euclidean geometry and temporal voting filters.
    """

    def __init__(self, stability_frames: int = 4):
        self.stability_frames = stability_frames
        self.history: deque = deque(maxlen=stability_frames)
        self.state = GestureState()

    @staticmethod
    def _euclidean_dist(pt1: Tuple[float, float, ...], pt2: Tuple[float, float, ...]) -> float:
        """Computes Euclidean distance between two points."""
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def compute_finger_states(self, hand: HandData) -> Dict[str, bool]:
        """
        Determines whether each finger is EXTENDED (True) or CURLED (False).
        Uses scale-invariant distance ratios from wrist to tip vs PIP/MCP.
        """
        lms = hand.landmarks_norm
        wrist = lms[0]
        palm = hand.palm_center_norm

        # Hand scale reference: distance from wrist (0) to middle MCP (9)
        hand_scale = max(0.05, self._euclidean_dist(wrist, lms[9]))

        # Helper to check if a finger (index, middle, ring, pinky) is extended
        def is_finger_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            dist_tip_wrist = self._euclidean_dist(lms[tip_idx], wrist)
            dist_pip_wrist = self._euclidean_dist(lms[pip_idx], wrist)
            dist_tip_palm = self._euclidean_dist(lms[tip_idx], palm)
            dist_mcp_palm = self._euclidean_dist(lms[mcp_idx], palm)

            # Finger is extended if tip is further from wrist than PIP
            # AND tip is clearly extended outward from the palm centroid
            is_dist_extended = dist_tip_wrist > (dist_pip_wrist * 1.10)
            is_away_from_palm = dist_tip_palm > (dist_mcp_palm * 1.05)
            return bool(is_dist_extended and is_away_from_palm)

        index_ext = is_finger_extended(8, 6, 5)
        middle_ext = is_finger_extended(12, 10, 9)
        ring_ext = is_finger_extended(16, 14, 13)
        pinky_ext = is_finger_extended(20, 18, 17)

        # Thumb extension detection:
        # Distance between thumb tip (4) and pinky MCP (17) or middle MCP (9)
        thumb_tip = lms[4]
        dist_thumb_pinky = self._euclidean_dist(thumb_tip, lms[17])
        dist_thumb_mcp = self._euclidean_dist(thumb_tip, lms[2])
        thumb_ext = bool(dist_thumb_pinky > (hand_scale * 0.95) and dist_thumb_mcp > (hand_scale * 0.35))

        return {
            "thumb": thumb_ext,
            "index": index_ext,
            "middle": middle_ext,
            "ring": ring_ext,
            "pinky": pinky_ext,
        }

    def _classify_raw_gesture(
        self, hand: HandData, fingers: Dict[str, bool]
    ) -> Tuple[str, float]:
        """
        Classifies the single-frame candidate gesture based on finger extension states
        and precise landmark geometry.
        """
        lms = hand.landmarks_norm
        wrist = lms[0]
        hand_scale = max(0.05, self._euclidean_dist(wrist, lms[9]))

        t = fingers["thumb"]
        i = fingers["index"]
        m = fingers["middle"]
        r = fingers["ring"]
        p = fingers["pinky"]

        # Number of extended 4 fingers (index, middle, ring, pinky)
        num_extended_four = sum([i, m, r, p])

        # -------------------------------------------------------------
        # 1. THUMBS UP: 4 fingers curled, thumb pointing upward
        # -------------------------------------------------------------
        if num_extended_four == 0:
            thumb_tip = lms[4]
            thumb_ip = lms[3]
            thumb_mcp = lms[2]
            # In image coords, y decreases as you move UP
            thumb_pointing_up = (thumb_tip[1] < thumb_ip[1]) and (thumb_tip[1] < thumb_mcp[1] - 0.03)
            # Thumb tip should be higher than index MCP
            thumb_above_knuckles = thumb_tip[1] < lms[5][1]

            if thumb_pointing_up and thumb_above_knuckles:
                confidence = 0.95
                return "THUMBS_UP", confidence

        # -------------------------------------------------------------
        # 2. CLOSED FIST: All 4 fingers curled inward
        # -------------------------------------------------------------
        if num_extended_four == 0:
            # Check if fingertips (8, 12, 16, 20) are curled close to palm center
            tip_dists = [self._euclidean_dist(lms[idx], hand.palm_center_norm) for idx in (8, 12, 16, 20)]
            avg_tip_dist = sum(tip_dists) / len(tip_dists)
            if avg_tip_dist < hand_scale * 0.95:
                confidence = 0.92
                return "CLOSED_FIST", confidence

        # -------------------------------------------------------------
        # 3. V-SIGN / PEACE: Index & Middle extended, Ring & Pinky curled
        # -------------------------------------------------------------
        if i and m and not r and not p:
            # Check separation between index tip (8) and middle tip (12)
            separation = self._euclidean_dist(lms[8], lms[12])
            confidence = 0.94 if separation > hand_scale * 0.15 else 0.85
            return "V_SIGN", confidence

        # -------------------------------------------------------------
        # 4. OPEN PALM: All 5 fingers extended, or at least 4 main fingers
        # -------------------------------------------------------------
        if num_extended_four == 4:
            confidence = 0.98 if t else 0.90
            return "OPEN_PALM", confidence

        if num_extended_four == 3 and t and (i and m):
            # Sometimes pinky or ring landmark glitches slightly, high confidence palm
            return "OPEN_PALM", 0.80

        return "NONE", 0.0

    def update(self, hand: Optional[HandData]) -> GestureState:
        """
        Updates the gesture recognition pipeline for a given hand.
        If hand is None (lost tracking), resets candidate state.
        """
        if hand is None:
            self.history.clear()
            self.state = GestureState(
                confirmed_gesture="NONE",
                raw_gesture="NONE",
                confidence=0.0,
                stability_count=0,
                is_stable=False,
            )
            return self.state

        # Compute current finger states
        fingers = self.compute_finger_states(hand)
        self.state.finger_states = fingers

        # Classify raw gesture in the current frame
        raw_gesture, confidence = self._classify_raw_gesture(hand, fingers)
        self.state.raw_gesture = raw_gesture
        self.state.confidence = confidence

        # Append to rolling history buffer
        self.history.append(raw_gesture)

        # Count occurrences of the candidate in recent history
        count = self.history.count(raw_gesture)
        self.state.stability_count = count

        # Debouncing rule: candidate must appear consistently across stability_frames
        if len(self.history) >= self.stability_frames and count >= self.stability_frames:
            self.state.confirmed_gesture = raw_gesture
            self.state.is_stable = (raw_gesture != "NONE")
        else:
            self.state.is_stable = False

        return self.state

    def reset(self) -> None:
        """Reset gesture history and state."""
        self.history.clear()
        self.state = GestureState()
