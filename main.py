"""
CV Fighter — Vision-Based Game Controller
Main Application Entry Point.

Controls existing browser fighting games using real-time hand gestures and movements.
WEBCAM -> MEDIAPIPE -> GESTURE RECOGNITION -> KEYBOARD INPUT -> BROWSER GAME
"""

import argparse
import logging
import sys
import time
from typing import Optional, Tuple
import cv2

# Project Modules
import config
from controller.keyboard_controller import KeyboardController
from controller.input_manager import InputManager
from utils.fps_counter import FPSCounter
from utils.hud_display import HUDDisplay
from vision.hand_detector import HandDetector, HandData
from vision.gesture_recognizer import GestureRecognizer
from vision.movement_detector import MovementDetector

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CVFighter")


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
        help="Start in TEST MODE (gestures visualized on HUD, keyboard emission disabled)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Start directly in LIVE GAME CONTROL MODE (sends real keystrokes to OS)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=config.DEBUG_MODE_DEFAULT,
        help="Enable on-screen debug diagnostic metrics",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config.CAMERA_WIDTH,
        help=f"Camera frame width (default: {config.CAMERA_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.CAMERA_HEIGHT,
        help=f"Camera frame height (default: {config.CAMERA_HEIGHT})",
    )
    return parser.parse_args()


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Attempts to open the specified camera device. If unavailable, probes
    alternate camera indices and raises a descriptive runtime error if none open.
    """
    logger.info(f"Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    # Set requested resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

    if not cap.isOpened():
        logger.warning(f"Could not open camera index {camera_index}. Probing indices 0, 1, 2...")
        for idx in [0, 1, 2]:
            if idx == camera_index:
                continue
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    logger.info(f"Successfully found working camera at index {idx}!")
                    test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    return test_cap
                test_cap.release()

        error_msg = (
            f"\n"
            f"===============================================================\n"
            f"ERROR: Unable to open webcam at index {camera_index}!\n"
            f"===============================================================\n"
            f"Troubleshooting Steps:\n"
            f"1. Ensure your laptop webcam is plugged in or enabled.\n"
            f"2. Ensure no other program (Zoom, Teams, Discord, Browser) is\n"
            f"   currently using the webcam.\n"
            f"3. Try passing a different camera index: python main.py --camera 1\n"
            f"4. On Windows 10/11, check Windows Privacy Settings -> Camera ->\n"
            f"   'Allow apps to access your camera' is turned ON.\n"
            f"==============================================================="
        )
        logger.error(error_msg)
        raise RuntimeError("No working webcam device could be opened.")

    return cap


def separate_hands(
    hands: list[HandData],
    assignment_mode: str = config.HAND_ASSIGNMENT_MODE,
) -> Tuple[Optional[HandData], Optional[HandData]]:
    """
    Separates detected hands into (MovementHand, AttackHand) based on screen
    position or MediaPipe classification.
    """
    if not hands:
        return None, None

    movement_hand: Optional[HandData] = None
    attack_hand: Optional[HandData] = None

    if assignment_mode == "screen_position":
        # Hand on the left half of mirrored view controls movement;
        # Hand on the right half controls attacks.
        for hand in hands:
            if hand.screen_side == "left" and movement_hand is None:
                movement_hand = hand
            elif hand.screen_side == "right" and attack_hand is None:
                attack_hand = hand
    else:
        # Based on MediaPipe's classified handedness
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

    # Determine initial test mode (live flag overrides test default)
    test_mode = False if args.live else args.test
    debug_mode = args.debug

    print("\n" + "=" * 64)
    print("  CV FIGHTER — Vision-Based Game Controller")
    print("=" * 64)
    print(f"  Mode:            {'TEST MODE (Safe Simulation)' if test_mode else 'LIVE GAME CONTROL'}")
    print(f"  Camera Index:    {args.camera}")
    print(f"  Resolution:      {args.width}x{args.height}")
    print(f"  Key Backend:     {config.KEYBOARD_BACKEND}")
    print("  Controls:")
    print("    [ESC]          Emergency Stop and Exit")
    print("    [T]            Toggle Test Mode vs Live Game Control")
    print("    [C]            Calibrate Neutral Anchor")
    print("    [D]            Toggle Debug Overlay")
    print("=" * 64 + "\n")

    # Initialize Core Subsystems
    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as e:
        sys.exit(1)

    hand_detector = HandDetector(
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

    window_title = "CV Fighter — Vision Game Controller"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, args.width, args.height)

    logger.info("CV Fighter controller initialized successfully. Running main loop...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to grab camera frame. Attempting to reconnect...")
                time.sleep(0.1)
                continue

            # 1. Process Vision & Hand Landmarks (mirrored for intuitive selfie perspective)
            frame, detected_hands = hand_detector.process(frame, mirror=config.MIRROR_VIEW)

            # 2. Separate Hands by Assigned Role
            movement_hand, attack_hand = separate_hands(detected_hands)

            # 3. Update Movement Tracking (Left Hand)
            movement_state = movement_detector.update(movement_hand)

            # 4. Update Gesture Recognition (Right Hand)
            gesture_state = gesture_recognizer.update(attack_hand)

            # 5. Dispatch Controls to Input Manager
            input_manager.update_movement(movement_state.active_directions)
            input_manager.update_action(gesture_state)

            # 6. Render Hand Skeletons
            if movement_hand:
                hand_detector.draw_skeleton(
                    frame,
                    movement_hand,
                    bone_color=(255, 180, 0),    # Cyan/Blue
                    joint_color=(255, 255, 255),
                    palm_color=(0, 255, 100),
                    draw_labels=debug_mode,
                )

            if attack_hand:
                hand_detector.draw_skeleton(
                    frame,
                    attack_hand,
                    bone_color=(0, 180, 255),    # Orange/Gold
                    joint_color=(255, 255, 255),
                    palm_color=(0, 200, 255),
                    draw_labels=debug_mode,
                )

            # 7. Update Performance & Render HUD Overlay
            fps_counter.update()
            fps_str = fps_counter.get_fps_str()

            hud.render(
                frame=frame,
                movement_state=movement_state,
                gesture_state=gesture_state,
                input_mgr=input_manager,
                fps_str=fps_str,
                test_mode=input_manager.test_mode,
                debug_mode=debug_mode,
                movement_hand=movement_hand,
                attack_hand=attack_hand,
            )

            # 8. Display Canvas
            cv2.imshow(window_title, frame)

            # 9. Handle Hotkeys
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                logger.info("Exit requested by user (ESC / Q). Shutting down...")
                break
            elif key in (ord("t"), ord("T")):
                new_state = input_manager.toggle_test_mode()
                logger.info(f"Mode toggled: {'TEST MODE' if new_state else 'LIVE CONTROL'}")
            elif key in (ord("c"), ord("C")):
                anchor = movement_detector.calibrate(movement_hand)
                hud.trigger_calibration_notice()
                logger.info(f"Calibrated neutral anchor to: ({anchor[0]:.2f}, {anchor[1]:.2f})")
            elif key in (ord("d"), ord("D")):
                debug_mode = not debug_mode
                logger.info(f"Debug overlay: {'ENABLED' if debug_mode else 'DISABLED'}")

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt intercepted! Exiting cleanly...")

    finally:
        # Guarantee Safe Cleanup
        logger.info("Executing safety cleanup: releasing all keys and camera resources...")
        input_manager.cleanup()
        hand_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("CV Fighter stopped safely.")


if __name__ == "__main__":
    main()
