"""
CV Fighter — Configuration Module
Contains all user-adjustable parameters for camera, computer vision,
movement deadzones, gesture recognition, and keyboard mappings.
"""

# ==============================================================================
# 1. CAMERA SETTINGS
# ==============================================================================
# Default webcam index (0 is usually the integrated laptop webcam)
CAMERA_INDEX = 0

# Video capture dimensions (1280x720 recommended for crisp landmark tracking)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Target frame rate for video capture and processing loop
TARGET_FPS = 30

# Mirror the camera horizontally (True gives a natural selfie-mirror view)
MIRROR_VIEW = True


# ==============================================================================
# 2. TWO-HAND ASSIGNMENT SETTINGS
# ==============================================================================
# Hand assignment method:
#   "screen_position": Hand on the left side of the screen controls movement;
#                      Hand on the right side of the screen controls attacks.
#                      (Highly robust, immune to MediaPipe left/right hand flipping)
#   "mediapipe":       Uses MediaPipe's classified handedness.
HAND_ASSIGNMENT_MODE = "screen_position"

# Which role belongs to which hand / screen side
MOVEMENT_HAND = "left"   # Options: "left", "right"
ATTACK_HAND = "right"    # Options: "left", "right"


# ==============================================================================
# 3. MOVEMENT DETECTION & DEADZONES (LEFT HAND)
# ==============================================================================
# Movement operates like a virtual analog stick with a neutral deadzone center.
# Deadzones are expressed as a fraction of the screen dimensions [0.0 - 1.0].
MOVEMENT_DEADZONE_X = 0.08   # Horizontal threshold (approx 100px at 1280 wide)
MOVEMENT_DEADZONE_Y = 0.08   # Vertical threshold (approx 60px at 720 high)

# Position smoothing filter (Exponential Moving Average, 0.0 to 1.0).
# Higher = more responsive, lower = smoother / less jitter.
MOVEMENT_SMOOTHING = 0.65

# Default normalized neutral anchor (X, Y) for the movement hand.
# Can be recalibrated on-the-fly by pressing 'C'.
DEFAULT_MOVEMENT_ANCHOR_X = 0.25   # Centered in the left half of the screen
DEFAULT_MOVEMENT_ANCHOR_Y = 0.55   # Slightly below vertical center


# ==============================================================================
# 4. GESTURE RECOGNITION & STABILITY (RIGHT HAND)
# ==============================================================================
# Number of consecutive frames a gesture must be detected before activating.
# Prevents single-frame detection noise from triggering accidental attacks.
GESTURE_STABILITY_FRAMES = 4

# Minimum cooldown period (in seconds) between successive attacks.
# Prevents spamming and ensures stable combo execution.
ATTACK_COOLDOWN_SEC = 0.35

# Duration (in seconds) that an attack key is held down before releasing.
# Guarantees the browser/game registers the key press cleanly.
ATTACK_KEY_TAP_DURATION_SEC = 0.05

# Minimum confidence thresholds for MediaPipe Hands
MIN_DETECTION_CONFIDENCE = 0.65
MIN_TRACKING_CONFIDENCE = 0.60


# ==============================================================================
# 5. KEYBOARD MAPPINGS
# ==============================================================================
# Mappings for movement (continuous KEY HOLD behavior)
# and actions (single KEY PRESS / TAP behavior).
KEY_MAPPINGS = {
    # Player 1 Movement
    "UP": "w",         # Jump / Move Up
    "LEFT": "a",       # Move Left
    "DOWN": "s",       # Crouch / Move Down
    "RIGHT": "d",      # Move Right

    # Player 1 Actions / Attacks
    "ATTACK_1": "y",   # Open Palm Gesture
    "ATTACK_2": "h",   # Closed Fist Gesture
    "ATTACK_3": "g",   # Two Fingers / V-Sign Gesture
    "ATTACK_4": "j",   # Thumbs-Up Gesture
}

# Human-readable labels for each gesture to display on the HUD
GESTURE_NAMES = {
    "NONE": "Neutral / Idle",
    "OPEN_PALM": "Open Palm",
    "CLOSED_FIST": "Closed Fist",
    "V_SIGN": "V-Sign (Peace)",
    "THUMBS_UP": "Thumbs Up",
}

# Mapping of gesture names to corresponding action keys
GESTURE_TO_ACTION = {
    "OPEN_PALM": "ATTACK_1",
    "CLOSED_FIST": "ATTACK_2",
    "V_SIGN": "ATTACK_3",
    "THUMBS_UP": "ATTACK_4",
}


# ==============================================================================
# 6. OPERATING MODES & SAFETY DEFAULTS
# ==============================================================================
# When TEST_MODE is True, all computer vision and gestures are detected and
# rendered on screen, but NO keyboard inputs are sent to the OS.
# Toggle anytime during runtime by pressing 'T'.
TEST_MODE_DEFAULT = True

# When DEBUG_MODE is True, detailed landmark coordinates, finger states,
# confidence metrics, and internal timers are rendered on screen.
# Toggle anytime during runtime by pressing 'D'.
DEBUG_MODE_DEFAULT = False

# Keyboard driver backend ("pynput" or "pyautogui")
KEYBOARD_BACKEND = "pynput"


# ==============================================================================
# 7. VISUAL THEME & HUD COLORS (BGR Format for OpenCV)
# ==============================================================================
COLORS = {
    "PRIMARY": (255, 180, 0),        # Cyan / Electric Blue
    "SECONDARY": (0, 255, 255),      # Bright Yellow
    "ACCENT": (0, 128, 255),         # Vivid Orange
    "SUCCESS": (50, 220, 50),        # High-visibility Green
    "WARNING": (0, 165, 255),        # Amber Warning
    "DANGER": (50, 50, 240),         # Red / Stop
    "NEUTRAL": (180, 180, 180),      # Light Gray
    "DARK_BG": (20, 20, 25),         # Semi-dark overlay tint
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "DEADZONE": (80, 80, 80),        # Deadzone bounding outline
    "ACTIVE_ZONE": (0, 220, 100),    # Active directional fill
}
