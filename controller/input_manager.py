"""
Input Manager Module for CV Fighter
Orchestrates high-level game controls:
- Continuous KEY HOLD for movement (W, A, S, D)
- Single KEY PRESS / TAP for actions (Y, H, G, J) with debouncing & cooldown
- TEST MODE toggle (safe simulation without emitting OS keystrokes)
- Emergency Stop and safe release
"""

import time
import logging
from typing import Dict, Set, Optional, Tuple
from controller.keyboard_controller import KeyboardController
from vision.gesture_recognizer import GestureState

logger = logging.getLogger(__name__)


class InputManager:
    """
    Translates computer vision states into physical or simulated game keystrokes.
    Enforces the single-tap rule for attacks and key-hold rule for movements.
    """

    def __init__(
        self,
        keyboard: KeyboardController,
        key_mappings: Dict[str, str],
        gesture_to_action: Dict[str, str],
        attack_cooldown_sec: float = 0.35,
        attack_tap_duration_sec: float = 0.05,
        test_mode: bool = True,
    ):
        self.keyboard = keyboard
        self.key_mappings = key_mappings
        self.gesture_to_action = gesture_to_action
        self.attack_cooldown_sec = attack_cooldown_sec
        self.attack_tap_duration_sec = attack_tap_duration_sec
        self.test_mode = test_mode

        # Movement tracking
        self.held_movement_keys: Set[str] = set()

        # Action / Attack tracking
        self.last_attack_time: float = 0.0
        self.active_action_gesture: str = "NONE"
        self.action_latched: bool = False  # True if current continuous gesture already fired

        # HUD feedback tracking
        self.last_triggered_action: Optional[str] = None
        self.last_triggered_key: Optional[str] = None
        self.last_triggered_time: float = 0.0
        self.feedback_duration: float = 0.5  # Seconds visual indicator remains highlighted

        self.is_active: bool = True

    def toggle_test_mode(self) -> bool:
        """
        Toggles between TEST MODE and LIVE GAME CONTROL MODE.
        Releases all held keys when toggling for safety.
        """
        self.keyboard.release_all()
        self.held_movement_keys.clear()
        self.test_mode = not self.test_mode
        logger.info(f"InputManager: TEST MODE is now {'ENABLED' if self.test_mode else 'DISABLED (LIVE)'}")
        return self.test_mode

    def update_movement(self, active_directions: Set[str]) -> Tuple[Set[str], Set[str]]:
        """
        Processes movement directions and updates KEY HOLD states.
        Returns: (keys_held, directions_active)
        """
        if not self.is_active:
            return set(), set()

        # Translate directions to keys (e.g. "LEFT" -> "a")
        target_keys = {
            self.key_mappings[d]
            for d in active_directions
            if d in self.key_mappings
        }

        keys_to_release = self.held_movement_keys - target_keys
        keys_to_press = target_keys - self.held_movement_keys

        # Apply key releases
        for k in keys_to_release:
            if not self.test_mode:
                self.keyboard.release(k)

        # Apply key presses (Key Down once, kept held)
        for k in keys_to_press:
            if not self.test_mode:
                self.keyboard.press(k)

        self.held_movement_keys = target_keys
        return self.held_movement_keys, active_directions

    def update_action(self, gesture_state: GestureState) -> Optional[str]:
        """
        Processes combat gesture states and triggers single-tap attacks.
        Enforces:
          1. Debouncing (gesture must be confirmed stable).
          2. Single-shot latch (holding a fist will fire ONCE until relaxed).
          3. Cooldown window (prevents rapid spamming).
        Returns the triggered action name (e.g. 'ATTACK_2') or None.
        """
        if not self.is_active:
            return None

        current_gesture = gesture_state.confirmed_gesture
        now = time.time()

        # If hand returned to neutral/idle or gesture broke, unlatch immediately
        if current_gesture == "NONE" or not gesture_state.is_stable:
            self.action_latched = False
            self.active_action_gesture = "NONE"
            return None

        # If this exact gesture is already active and latched, DO NOT retrigger
        if self.action_latched and (self.active_action_gesture == current_gesture):
            return None

        # Check if the gesture maps to an attack action
        if current_gesture in self.gesture_to_action:
            # Check attack cooldown
            if (now - self.last_attack_time) >= self.attack_cooldown_sec:
                action_name = self.gesture_to_action[current_gesture]
                key_to_press = self.key_mappings.get(action_name)

                if key_to_press:
                    if not self.test_mode:
                        # Send single atomic key tap
                        self.keyboard.tap(key_to_press, self.attack_tap_duration_sec)

                    # Update state and latch
                    self.last_attack_time = now
                    self.active_action_gesture = current_gesture
                    self.action_latched = True

                    # Record visual feedback for HUD
                    self.last_triggered_action = action_name
                    self.last_triggered_key = key_to_press
                    self.last_triggered_time = now

                    logger.debug(f"Triggered action '{action_name}' (Key: {key_to_press}) via {current_gesture}")
                    return action_name

        return None

    def get_action_feedback(self) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Returns (action_name, key, is_fresh_trigger) for HUD rendering.
        """
        now = time.time()
        is_active = (now - self.last_triggered_time) < self.feedback_duration
        if is_active:
            return self.last_triggered_action, self.last_triggered_key, True
        return None, None, False

    def emergency_stop(self) -> None:
        """Immediately release all keys and halt input."""
        self.is_active = False
        self.keyboard.release_all()
        self.held_movement_keys.clear()
        logger.warning("InputManager: Emergency Stop triggered! All keys released.")

    def cleanup(self) -> None:
        """Clean shutdown releasing all resources and keys."""
        self.is_active = False
        self.keyboard.release_all()
        self.held_movement_keys.clear()
