"""
Movement Detector Module for CV Fighter
Implements a virtual joystick / D-Pad with neutral anchor point,
exponential smoothing, configurable deadzones, and runtime calibration.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
import numpy as np
from vision.hand_detector import HandData


@dataclass
class MovementState:
    """Current state of movement tracking."""
    # Current active movement directions: e.g. {"LEFT"}, {"UP", "RIGHT"}, or empty (NEUTRAL)
    active_directions: Set[str] = field(default_factory=set)
    # Primary single direction for clean display
    primary_direction: str = "NEUTRAL"
    # Smoothed normalized palm center (X, Y)
    current_pos: Tuple[float, float] = (0.25, 0.55)
    # Calibrated neutral anchor (X, Y)
    anchor_pos: Tuple[float, float] = (0.25, 0.55)
    # Normalized offset delta: (dx, dy) = (current - anchor)
    offset: Tuple[float, float] = (0.0, 0.0)
    # True if the movement hand is actively detected in the frame
    hand_detected: bool = False


class MovementDetector:
    """
    Tracks hand position relative to a calibrated neutral center to detect
    continuous directional movements (UP, DOWN, LEFT, RIGHT).
    """

    def __init__(
        self,
        anchor_x: float = 0.25,
        anchor_y: float = 0.55,
        deadzone_x: float = 0.08,
        deadzone_y: float = 0.08,
        smoothing: float = 0.65,
    ):
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.deadzone_x = deadzone_x
        self.deadzone_y = deadzone_y
        self.smoothing = smoothing

        # Smoothed position filter state
        self._smoothed_x: Optional[float] = None
        self._smoothed_y: Optional[float] = None

        self.state = MovementState(
            anchor_pos=(anchor_x, anchor_y),
            current_pos=(anchor_x, anchor_y),
        )

    def calibrate(self, hand: Optional[HandData] = None) -> Tuple[float, float]:
        """
        Calibrate the neutral anchor to the hand's current position.
        If no hand is present, resets to default coordinates.
        """
        if hand is not None:
            self.anchor_x, self.anchor_y = hand.palm_center_norm
        self._smoothed_x = self.anchor_x
        self._smoothed_y = self.anchor_y
        self.state.anchor_pos = (self.anchor_x, self.anchor_y)
        return (self.anchor_x, self.anchor_y)

    def update(self, hand: Optional[HandData]) -> MovementState:
        """
        Processes the movement hand in the current frame and computes active directions.
        """
        if hand is None:
            # Movement hand lost; return to neutral safely
            self._smoothed_x = None
            self._smoothed_y = None
            self.state.active_directions.clear()
            self.state.primary_direction = "NEUTRAL"
            self.state.offset = (0.0, 0.0)
            self.state.hand_detected = False
            return self.state

        raw_x, raw_y = hand.palm_center_norm

        # Exponential Moving Average (EMA) smoothing
        if self._smoothed_x is None:
            self._smoothed_x = raw_x
            self._smoothed_y = raw_y
        else:
            self._smoothed_x = (self.smoothing * raw_x) + ((1.0 - self.smoothing) * self._smoothed_x)
            self._smoothed_y = (self.smoothing * raw_y) + ((1.0 - self.smoothing) * self._smoothed_y)

        # Compute offset relative to anchor
        dx = self._smoothed_x - self.anchor_x
        dy = self._smoothed_y - self.anchor_y

        directions: Set[str] = set()

        # Horizontal threshold evaluation
        if dx < -self.deadzone_x:
            directions.add("LEFT")
        elif dx > self.deadzone_x:
            directions.add("RIGHT")

        # Vertical threshold evaluation (negative dy is UP in screen coordinates)
        if dy < -self.deadzone_y:
            directions.add("UP")
        elif dy > self.deadzone_y:
            directions.add("DOWN")

        # Determine dominant primary direction for clear HUD display
        if not directions:
            primary = "NEUTRAL"
        elif len(directions) == 1:
            primary = next(iter(directions))
        else:
            # Select axis with larger relative displacement past its deadzone
            norm_excess_x = (abs(dx) - self.deadzone_x) / self.deadzone_x
            norm_excess_y = (abs(dy) - self.deadzone_y) / self.deadzone_y
            if norm_excess_x >= norm_excess_y:
                primary = "LEFT" if dx < 0 else "RIGHT"
            else:
                primary = "UP" if dy < 0 else "DOWN"

        self.state.active_directions = directions
        self.state.primary_direction = primary
        self.state.current_pos = (self._smoothed_x, self._smoothed_y)
        self.state.anchor_pos = (self.anchor_x, self.anchor_y)
        self.state.offset = (dx, dy)
        self.state.hand_detected = True

        return self.state

    def reset(self) -> None:
        """Reset state tracking."""
        self._smoothed_x = None
        self._smoothed_y = None
        self.state = MovementState(
            anchor_pos=(self.anchor_x, self.anchor_y),
            current_pos=(self.anchor_x, self.anchor_y),
        )
