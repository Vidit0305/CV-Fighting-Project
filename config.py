"""
CV Fighter — Configuration Module
Contains all user-adjustable parameters for camera, computer vision,
movement deadzones, gesture recognition, and keyboard mappings.
"""

# ==============================================================================
# 1. CAMERA & DISPLAY SETTINGS
# ==============================================================================
# Default webcam index (0 is usually the integrated laptop webcam)
CAMERA_INDEX = 0

# Video capture dimensions
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Target display screen dimensions for fullscreen edge-to-edge fill
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

# Launch in Fullscreen borderless mode by default
FULLSCREEN_DEFAULT = True

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
MOVEMENT_DEADZONE_X = 0.08   # Horizontal threshold
MOVEMENT_DEADZONE_Y = 0.08   # Vertical threshold

# Position smoothing filter (Exponential Moving Average, 0.0 to 1.0).
MOVEMENT_SMOOTHING = 0.65

# Default normalized neutral anchor (X, Y) for the movement hand.
DEFAULT_MOVEMENT_ANCHOR_X = 0.25   # Centered in the left half of the screen
DEFAULT_MOVEMENT_ANCHOR_Y = 0.55   # Slightly below vertical center


# ==============================================================================
# 4. GESTURE RECOGNITION & STABILITY (RIGHT HAND)
# ==============================================================================
# Number of consecutive frames a gesture must be detected before activating.
GESTURE_STABILITY_FRAMES = 4

# Minimum cooldown period (in seconds) between successive attacks.
ATTACK_COOLDOWN_SEC = 0.35

# Duration (in seconds) that an attack key is held down before releasing.
ATTACK_KEY_TAP_DURATION_SEC = 0.05

# MediaPipe model complexity: 0 = Lite (ultra-fast, real-time CPU), 1 = Full
MODEL_COMPLEXITY = 0

# Minimum confidence thresholds for MediaPipe Hands
MIN_DETECTION_CONFIDENCE = 0.55
MIN_TRACKING_CONFIDENCE = 0.55


# ==============================================================================
# 5. KEYBOARD MAPPINGS
# ==============================================================================
KEY_MAPPINGS = {
    # Player 1 Movement (Continuous KEY HOLD)
    "UP": "w",         # Jump / Move Up
    "LEFT": "a",       # Move Left
    "DOWN": "s",       # Crouch / Move Down
    "RIGHT": "d",      # Move Right

    # Player 1 Actions / Attacks (Single KEY PRESS / TAP)
    "ATTACK_1": "y",   # Open Palm Gesture
    "ATTACK_2": "h",   # Closed Fist Gesture
    "ATTACK_3": "g",   # Two Fingers / V-Sign Gesture
    "ATTACK_4": "j",   # Thumbs-Up Gesture
}

GESTURE_NAMES = {
    "NONE": "Neutral",
    "OPEN_PALM": "Open Palm",
    "CLOSED_FIST": "Closed Fist",
    "V_SIGN": "V-Sign (Peace)",
    "THUMBS_UP": "Thumbs Up",
}

GESTURE_TO_ACTION = {
    "OPEN_PALM": "ATTACK_1",
    "CLOSED_FIST": "ATTACK_2",
    "V_SIGN": "ATTACK_3",
    "THUMBS_UP": "ATTACK_4",
}


# ==============================================================================
# 6. OPERATING MODES & SAFETY DEFAULTS
# ==============================================================================
TEST_MODE_DEFAULT = True
DEBUG_MODE_DEFAULT = False
KEYBOARD_BACKEND = "pynput"


# ==============================================================================
# 7. VISUAL THEME & HUD COLORS (BGR Format for OpenCV)
# ==============================================================================
COLORS = {
    "PRIMARY": (255, 190, 0),        # Electric Cyan / Blue
    "SECONDARY": (0, 220, 255),      # Neon Amber / Yellow
    "ACCENT": (0, 140, 255),         # Vivid Orange
    "SUCCESS": (80, 230, 80),        # High-visibility Emerald Green
    "WARNING": (0, 180, 255),        # Amber Warning
    "DANGER": (60, 60, 250),         # Red / Alert
    "NEUTRAL": (170, 170, 170),      # Crisp Light Gray
    "DARK_BG": (18, 18, 22),         # Sleek Dark Tint
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
}
