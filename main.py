"""
CV Fighter — Vision-Based Game Controller
Main Application Entry Point.

Controls existing browser fighting games using real-time hand gestures and movements.
Features:
- LIVE GAME CONTROL by default (keys immediately sent to browser game)
- Global hotkey listener (T, C, ESC work even when browser window is focused)
- Edge-to-edge scaling with zero white/gray sidebars
- Threaded high-FPS camera capture for silky-smooth video playback
- Asynchronous fail-safe keyboard control (continuous key-hold + single-tap attacks)
"""

import argparse
import logging
import sys
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from pynput import keyboard as pynput_kb

# Project Modules
import config
from controller.keyboard_controller import KeyboardController
from controller.input_manager import InputManager
from utils.fps_counter import FPSCounter
from utils.hud_display import HUDDisplay
from vision.hand_detector import HandDetector, HandData
from vision.gesture_recognizer import GestureRecognizer
from vision.movement_detector import MovementDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CVFighter")


class ThreadedCamera:
    """
    Dedicated threaded camera reader.
    Captures frames asynchronously in a background thread to prevent
    inference latency from blocking the camera sensor and to eliminate frame stutter.
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.cap = cv2.VideoCapture(camera_index)

        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera at index {camera_index}")

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def fit_to_screen(frame: np.ndarray, target_w: int = 1920, target_h: int = 1080) -> np.ndarray:
    """
    Scales and crops the camera frame edge-to-edge to eliminate white or black borders.
    Preserves natural aspect ratio by cropping any excess rather than stretching.
    """
    h, w = frame.shape[:2]
    target_aspect = target_w / target_h
    current_aspect = w / h

    if abs(current_aspect - target_aspect) > 0.02:
        if current_aspect < target_aspect:
            new_h = int(w / target_aspect)
            y_start = max(0, (h - new_h) // 2)
            frame = frame[y_start : y_start + new_h, :]
        else:
            new_w = int(h * target_aspect)
            x_start = max(0, (w - new_w) // 2)
            frame = frame[:, x_start : x_start + new_w]

    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def parse_arguments() -> argparse.Namespace:
    """Parse optional command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CV Fighter — Vision-Based Game Controller for Browser Fighting Games"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help=f"Webcam device index (default: {config.CAMERA_INDEX})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=config.TEST_MODE_DEFAULT,
        help="Start in TEST MODE (safe simulation with keyboard disabled)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Start directly in LIVE GAME CONTROL MODE (default)",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=config.FULLSCREEN_DEFAULT,
        help="Launch in fullscreen mode",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=config.DEBUG_MODE_DEFAULT,
        help="Enable on-screen debug diagnostic metrics",
    )
    return parser.parse_args()


def separate_hands(
    hands: list[HandData],
    assignment_mode: str = config.HAND_ASSIGNMENT_MODE,
) -> Tuple[Optional[HandData], Optional[HandData]]:
    """Separates detected hands into (MovementHand, AttackHand)."""
    if not hands:
        return None, None

    movement_hand: Optional[HandData] = None
    attack_hand: Optional[HandData] = None

    if assignment_mode == "screen_position":
        for hand in hands:
            if hand.screen_side == "left" and movement_hand is None:
                movement_hand = hand
            elif hand.screen_side == "right" and attack_hand is None:
                attack_hand = hand
    else:
        for hand in hands:
            hand_label = hand.mp_handedness.lower()
            if hand_label == config.MOVEMENT_HAND.lower() and movement_hand is None:
                movement_hand = hand
            elif hand_label == config.ATTACK_HAND.lower() and attack_hand is None:
                attack_hand = hand

    return movement_hand, attack_hand


def main() -> None:
    """Main application loop."""
    args = parse_arguments()

    test_mode = args.test if not args.live else False
    debug_mode = args.debug
    is_fullscreen = args.fullscreen

    print("\n" + "=" * 64)
    print("  CV FIGHTER — Vision-Based Game Controller")
    print("=" * 64)
    print(f"  Mode:            {'TEST MODE (Safe Simulation)' if test_mode else 'LIVE GAME CONTROL (Keys Active!)'}")
    print(f"  Fullscreen:      {'ENABLED' if is_fullscreen else 'WINDOWED'}")
    print(f"  Tap Duration:    {config.ATTACK_KEY_TAP_DURATION_SEC * 1000:.0f}ms")
    print("  Controls:")
    print("    [ESC]          Emergency Stop & Exit")
    print("    [T]            Toggle Test Mode vs Live Game Control")
    print("    [C]            Calibrate Neutral Movement Anchor")
    print("    [F]            Toggle Fullscreen")
    print("    [D]            Toggle Debug Overlay")
    print("=" * 64 + "\n")

    # Initialize Threaded Camera
    try:
        camera = ThreadedCamera(
            camera_index=args.camera,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.TARGET_FPS,
        )
    except Exception as e:
        logger.error(f"Failed to open camera ({e}). Ensure webcam is plugged in.")
        sys.exit(1)

    # Initialize Subsystems
    hand_detector = HandDetector(
        model_complexity=config.MODEL_COMPLEXITY,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    movement_detector = MovementDetector(
        anchor_x=config.DEFAULT_MOVEMENT_ANCHOR_X,
        anchor_y=config.DEFAULT_MOVEMENT_ANCHOR_Y,
        deadzone_x=config.MOVEMENT_DEADZONE_X,
        deadzone_y=config.MOVEMENT_DEADZONE_Y,
        smoothing=config.MOVEMENT_SMOOTHING,
    )

    gesture_recognizer = GestureRecognizer(
        stability_frames=config.GESTURE_STABILITY_FRAMES
    )

    keyboard = KeyboardController(backend=config.KEYBOARD_BACKEND)

    input_manager = InputManager(
        keyboard=keyboard,
        key_mappings=config.KEY_MAPPINGS,
        gesture_to_action=config.GESTURE_TO_ACTION,
        attack_cooldown_sec=config.ATTACK_COOLDOWN_SEC,
        attack_tap_duration_sec=config.ATTACK_KEY_TAP_DURATION_SEC,
        test_mode=test_mode,
    )

    fps_counter = FPSCounter()
    hud = HUDDisplay()

    # Window configuration
    window_title = "CV Fighter — Vision Game Controller"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    if is_fullscreen:
        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_title, 960, 540)

    # State variables for hotkey handling
    exit_requested = threading.Event()
    current_movement_hand: list = [None]  # Container for thread-safe access

    # Global Hotkey Listener (works even when browser window is focused)
    def on_global_press(key):
        try:
            if key == pynput_kb.Key.esc:
                logger.info("Global [ESC] intercepted: Emergency Stop!")
                exit_requested.set()
            elif hasattr(key, "char") and key.char:
                c = key.char.lower()
                if c == "t":
                    st = input_manager.toggle_test_mode()
                    logger.info(f"Global [T] pressed: {'TEST MODE' if st else 'LIVE CONTROL'}")
                elif c == "c":
                    anchor = movement_detector.calibrate(current_movement_hand[0])
                    hud.trigger_calibration_notice()
                    logger.info(f"Global [C] pressed: Calibrated anchor to ({anchor[0]:.2f}, {anchor[1]:.2f})")
        except Exception:
            pass

    global_listener = pynput_kb.Listener(on_press=on_global_press)
    global_listener.daemon = True
    global_listener.start()

    logger.info("CV Fighter controller running. LIVE GAME CONTROL IS ACTIVE!")

    try:
        while not exit_requested.is_set():
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # 1. Scale & crop frame edge-to-edge
            screen_frame = fit_to_screen(
                frame, target_w=config.DISPLAY_WIDTH, target_h=config.DISPLAY_HEIGHT
            )

            # 2. Process MediaPipe Hand Detection on mirrored frame
            screen_frame, detected_hands = hand_detector.process(
                screen_frame, mirror=config.MIRROR_VIEW
            )

            # 3. Separate Hands by Assigned Role
            movement_hand, attack_hand = separate_hands(detected_hands)
            current_movement_hand[0] = movement_hand

            # 4. Update Movement Tracking (Left Hand)
            movement_state = movement_detector.update(movement_hand)

            # 5. Update Gesture Recognition (Right Hand)
            gesture_state = gesture_recognizer.update(attack_hand)

            # 6. Dispatch Game Controls (Sends real keystrokes when not in test mode)
            input_manager.update_movement(movement_state.active_directions)
            input_manager.update_action(gesture_state)

            # 7. Render Sleek Glowing Skeletons
            if movement_hand:
                hand_detector.draw_skeleton(
                    screen_frame,
                    movement_hand,
                    bone_color=config.COLORS["PRIMARY"],
                    joint_color=config.COLORS["WHITE"],
                    draw_labels=debug_mode,
                )

            if attack_hand:
                hand_detector.draw_skeleton(
                    screen_frame,
                    attack_hand,
                    bone_color=config.COLORS["SECONDARY"],
                    joint_color=config.COLORS["WHITE"],
                    draw_labels=debug_mode,
                )

            # 8. Update FPS and Render Minimalist Gaming HUD
            fps_counter.update()
            fps_str = fps_counter.get_fps_str()

            hud.render(
                frame=screen_frame,
                movement_state=movement_state,
                gesture_state=gesture_state,
                input_mgr=input_manager,
                fps_str=fps_str,
                test_mode=input_manager.test_mode,
                debug_mode=debug_mode,
                movement_hand=movement_hand,
                attack_hand=attack_hand,
            )

            # 9. Display Frame
            cv2.imshow(window_title, screen_frame)

            # 10. Process Local Window Hotkeys
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                logger.info("Exit requested by user. Shutting down...")
                break
            elif key in (ord("t"), ord("T")):
                new_state = input_manager.toggle_test_mode()
                logger.info(f"Mode toggled: {'TEST MODE' if new_state else 'LIVE CONTROL'}")
            elif key in (ord("c"), ord("C")):
                anchor = movement_detector.calibrate(movement_hand)
                hud.trigger_calibration_notice()
                logger.info(f"Calibrated anchor to: ({anchor[0]:.2f}, {anchor[1]:.2f})")
            elif key in (ord("f"), ord("F")):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_title, 960, 540)
                logger.info(f"Fullscreen: {'ENABLED' if is_fullscreen else 'WINDOWED'}")
            elif key in (ord("d"), ord("D")):
                debug_mode = not debug_mode
                logger.info(f"Debug overlay: {'ENABLED' if debug_mode else 'DISABLED'}")

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt intercepted! Exiting cleanly...")

    finally:
        logger.info("Executing safety cleanup: releasing all held keys...")
        input_manager.cleanup()
        hand_detector.close()
        camera.release()
        cv2.destroyAllWindows()
        logger.info("CV Fighter stopped safely.")


if __name__ == "__main__":
    main()
