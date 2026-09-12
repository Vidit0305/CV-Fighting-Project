"""
Automated Test Suite for CV Fighter
Verifies:
- KeyboardController state tracking & safety cleanup
- InputManager key-hold vs single-tap rules, debouncing & cooldowns
- MovementDetector virtual joystick math, deadzones & calibration
- GestureRecognizer geometric rules & temporal debouncing
"""

import sys
import time
import unittest
from typing import List, Tuple

import config
from controller.keyboard_controller import KeyboardController
from controller.input_manager import InputManager
from vision.gesture_recognizer import GestureRecognizer, GestureState
from vision.movement_detector import MovementDetector
from vision.hand_detector import HandData


class MockKeyboardDriver:
    """Mock driver simulating OS keyboard events for deterministic unit tests."""

    def __init__(self):
        self.pressed_keys = set()
        self.history = []

    def press(self, key):
        self.pressed_keys.add(key)
        self.history.append(("PRESS", key))

    def release(self, key):
        self.pressed_keys.discard(key)
        self.history.append(("RELEASE", key))

    def keyDown(self, key):
        self.press(key)

    def keyUp(self, key):
        self.release(key)


class TestKeyboardController(unittest.TestCase):
    """Verifies low-level keyboard tracking and cleanup."""

    def setUp(self):
        self.controller = KeyboardController()
        # Inject mock driver
        self.mock_driver = MockKeyboardDriver()
        self.controller._driver = self.mock_driver
        self.controller._use_pynput = True

    def tearDown(self):
        self.controller.release_all()

    def test_press_and_release(self):
        self.assertFalse(self.controller.is_pressed("a"))
        # First press returns True (newly pressed)
        self.assertTrue(self.controller.press("a"))
        self.assertTrue(self.controller.is_pressed("a"))
        # Repeated press while held returns False (no-op)
        self.assertFalse(self.controller.press("a"))

        # Release returns True
        self.assertTrue(self.controller.release("a"))
        self.assertFalse(self.controller.is_pressed("a"))
        # Repeated release returns False
        self.assertFalse(self.controller.release("a"))

    def test_release_all_cleanup(self):
        self.controller.press("w")
        self.controller.press("a")
        self.controller.press("s")
        self.controller.press("d")
        self.assertEqual(len(self.controller.get_held_keys()), 4)

        self.controller.release_all()
        self.assertEqual(len(self.controller.get_held_keys()), 0)
        self.assertFalse(self.controller.is_pressed("w"))
        self.assertFalse(self.controller.is_pressed("a"))


class TestInputManager(unittest.TestCase):
    """Verifies Key Hold vs Single Key Press behaviors."""

    def setUp(self):
        self.controller = KeyboardController()
        self.mock_driver = MockKeyboardDriver()
        self.controller._driver = self.mock_driver
        self.controller._use_pynput = True

        self.manager = InputManager(
            keyboard=self.controller,
            key_mappings=config.KEY_MAPPINGS,
            gesture_to_action=config.GESTURE_TO_ACTION,
            attack_cooldown_sec=0.2,
            attack_tap_duration_sec=0.01,
            test_mode=False,  # Test live dispatch
        )

    def tearDown(self):
        self.manager.cleanup()

    def test_movement_key_hold_behavior(self):
        """Moving left must press 'a' and keep it held until released."""
        # 1. Move LEFT
        self.manager.update_movement({"LEFT"})
        self.assertIn("a", self.controller.get_held_keys())
        self.assertEqual(self.mock_driver.history[-1], ("PRESS", "a"))

        # 2. Remain in LEFT (consecutive frames) -> must NOT re-press 'a'
        initial_history_len = len(self.mock_driver.history)
        self.manager.update_movement({"LEFT"})
        self.assertEqual(len(self.mock_driver.history), initial_history_len)
        self.assertIn("a", self.controller.get_held_keys())

        # 3. Change direction: LEFT -> RIGHT
        self.manager.update_movement({"RIGHT"})
        self.assertNotIn("a", self.controller.get_held_keys())
        self.assertIn("d", self.controller.get_held_keys())

        # 4. Return to NEUTRAL (empty set) -> releases 'd'
        self.manager.update_movement(set())
        self.assertNotIn("d", self.controller.get_held_keys())
        self.assertEqual(len(self.controller.get_held_keys()), 0)

    def test_action_single_press_behavior(self):
        """Holding a fist for multiple frames must trigger ONLY ONE key tap."""
        fist_state = GestureState(
            confirmed_gesture="CLOSED_FIST",
            confidence=0.95,
            is_stable=True,
        )

        # 1. First trigger -> Attack 2 ('h') fired
        action = self.manager.update_action(fist_state)
        self.assertEqual(action, "ATTACK_2")
        self.assertTrue(self.manager.action_latched)

        # 2. Fist remains held for several frames -> NO new attack triggers
        for _ in range(5):
            action_repeat = self.manager.update_action(fist_state)
            self.assertIsNone(action_repeat)

        # 3. Fist is released (neutral/none)
        neutral_state = GestureState(confirmed_gesture="NONE", is_stable=False)
        self.manager.update_action(neutral_state)
        self.assertFalse(self.manager.action_latched)

        # 4. Wait for cooldown
        time.sleep(0.25)

        # 5. Fist appears again -> Attack 2 ('h') fires again
        action2 = self.manager.update_action(fist_state)
        self.assertEqual(action2, "ATTACK_2")


class TestMovementDetector(unittest.TestCase):
    """Verifies D-pad deadzone math and position filtering."""

    def setUp(self):
        self.detector = MovementDetector(
            anchor_x=0.5,
            anchor_y=0.5,
            deadzone_x=0.1,
            deadzone_y=0.1,
            smoothing=1.0,  # Disable smoothing for exact test coordinates
        )

    def test_neutral_zone(self):
        # Hand directly at anchor
        hand = HandData(palm_center_norm=(0.5, 0.5))
        state = self.detector.update(hand)
        self.assertEqual(len(state.active_directions), 0)
        self.assertEqual(state.primary_direction, "NEUTRAL")

    def test_directional_detection(self):
        # Move Left: x < 0.5 - 0.1 = 0.4
        hand_left = HandData(palm_center_norm=(0.35, 0.5))
        state = self.detector.update(hand_left)
        self.assertIn("LEFT", state.active_directions)
        self.assertEqual(state.primary_direction, "LEFT")

        # Move Right: x > 0.5 + 0.1 = 0.6
        hand_right = HandData(palm_center_norm=(0.65, 0.5))
        state = self.detector.update(hand_right)
        self.assertIn("RIGHT", state.active_directions)
        self.assertEqual(state.primary_direction, "RIGHT")

        # Move Up: y < 0.5 - 0.1 = 0.4
        hand_up = HandData(palm_center_norm=(0.5, 0.35))
        state = self.detector.update(hand_up)
        self.assertIn("UP", state.active_directions)
        self.assertEqual(state.primary_direction, "UP")

        # Move Down: y > 0.5 + 0.1 = 0.6
        hand_down = HandData(palm_center_norm=(0.5, 0.65))
        state = self.detector.update(hand_down)
        self.assertIn("DOWN", state.active_directions)
        self.assertEqual(state.primary_direction, "DOWN")

    def test_calibration(self):
        hand = HandData(palm_center_norm=(0.3, 0.4))
        anchor = self.detector.calibrate(hand)
        self.assertEqual(anchor, (0.3, 0.4))
        self.assertEqual(self.detector.anchor_x, 0.3)
        self.assertEqual(self.detector.anchor_y, 0.4)


class TestGestureRecognizer(unittest.TestCase):
    """Verifies gesture classification and debouncing."""

    def setUp(self):
        self.recognizer = GestureRecognizer(stability_frames=3)

    def _create_mock_hand(self, finger_extensions: Tuple[bool, bool, bool, bool, bool]) -> HandData:
        """
        Creates synthetic hand landmarks with specified finger extensions.
        Extensions: (thumb, index, middle, ring, pinky)
        """
        # Wrist at (0.5, 0.8)
        wrist = (0.5, 0.8, 0.0)
        lms = [wrist]

        thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = finger_extensions

        # Thumb (1-4)
        if thumb_ext:
            lms.extend([(0.42, 0.75, 0), (0.35, 0.70, 0), (0.30, 0.62, 0), (0.26, 0.55, 0)])
        else:
            lms.extend([(0.45, 0.75, 0), (0.45, 0.70, 0), (0.46, 0.68, 0), (0.48, 0.65, 0)])

        # Knuckle MCP base y ~ 0.65
        # Index (5-8)
        if index_ext:
            lms.extend([(0.42, 0.65, 0), (0.41, 0.52, 0), (0.40, 0.40, 0), (0.40, 0.28, 0)])
        else:
            lms.extend([(0.42, 0.65, 0), (0.42, 0.60, 0), (0.43, 0.66, 0), (0.44, 0.70, 0)])

        # Middle (9-12)
        if middle_ext:
            lms.extend([(0.50, 0.64, 0), (0.50, 0.50, 0), (0.50, 0.38, 0), (0.50, 0.25, 0)])
        else:
            lms.extend([(0.50, 0.64, 0), (0.50, 0.59, 0), (0.50, 0.65, 0), (0.50, 0.70, 0)])

        # Ring (13-16)
        if ring_ext:
            lms.extend([(0.58, 0.65, 0), (0.59, 0.52, 0), (0.60, 0.40, 0), (0.60, 0.28, 0)])
        else:
            lms.extend([(0.58, 0.65, 0), (0.58, 0.60, 0), (0.57, 0.66, 0), (0.56, 0.70, 0)])

        # Pinky (17-20)
        if pinky_ext:
            lms.extend([(0.65, 0.68, 0), (0.67, 0.56, 0), (0.68, 0.46, 0), (0.69, 0.35, 0)])
        else:
            lms.extend([(0.65, 0.68, 0), (0.65, 0.64, 0), (0.64, 0.68, 0), (0.63, 0.71, 0)])

        palm_center = (0.5, 0.68)
        return HandData(
            landmarks_norm=lms,
            palm_center_norm=palm_center,
            wrist_norm=(0.5, 0.8),
        )

    def test_closed_fist_debouncing(self):
        fist_hand = self._create_mock_hand((False, False, False, False, False))

        # Frame 1: Detected candidate, but not yet stable (1/3)
        state1 = self.recognizer.update(fist_hand)
        self.assertEqual(state1.raw_gesture, "CLOSED_FIST")
        self.assertFalse(state1.is_stable)

        # Frame 2: Still not stable (2/3)
        state2 = self.recognizer.update(fist_hand)
        self.assertFalse(state2.is_stable)

        # Frame 3: Confirmed stable! (3/3)
        state3 = self.recognizer.update(fist_hand)
        self.assertEqual(state3.confirmed_gesture, "CLOSED_FIST")
        self.assertTrue(state3.is_stable)

    def test_open_palm_classification(self):
        palm_hand = self._create_mock_hand((True, True, True, True, True))
        for _ in range(3):
            state = self.recognizer.update(palm_hand)
        self.assertEqual(state.confirmed_gesture, "OPEN_PALM")
        self.assertTrue(state.is_stable)

    def test_v_sign_classification(self):
        v_hand = self._create_mock_hand((False, True, True, False, False))
        for _ in range(3):
            state = self.recognizer.update(v_hand)
        self.assertEqual(state.confirmed_gesture, "V_SIGN")
        self.assertTrue(state.is_stable)


if __name__ == "__main__":
    unittest.main()
