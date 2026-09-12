# 🥊 CV Fighter — Vision-Based Game Controller

A high-performance, real-time **Computer Vision Keyboard Controller** that lets you control an existing browser-based fighting game using your laptop webcam and natural two-hand gestures.

> **Note**: This is **NOT** a new game and **NOT** a website or web application. It is a local background Computer Vision controller that captures hand movements through your webcam and translates them into actual physical OS keyboard inputs (`W`, `A`, `S`, `D`, `Y`, `H`, `G`, `J`) to play your existing browser fighting game.

---

## ⚡ Pipeline Architecture

```
Webcam Video (30 FPS)
         │
         ▼
OpenCV (Horizontal Mirror Flip)
         │
         ▼
MediaPipe Hands (21 3D Landmarks)
         │
         ├──────────────────────────────────────────────┐
         ▼                                              ▼
LEFT HAND (Screen Left)                        RIGHT HAND (Screen Right)
Movement Detection (Virtual D-Pad)             Gesture Recognition (Combat Actions)
  • Neutral Anchor with Deadzones                • Open Palm   ──► Attack 1 (Y)
  • Move Left   ──► Hold 'A'                     • Closed Fist ──► Attack 2 (H)
  • Move Right  ──► Hold 'D'                     • V-Sign      ──► Attack 3 (G)
  • Move Up     ──► Hold 'W'                     • Thumbs Up   ──► Attack 4 (J)
  • Move Down   ──► Hold 'S'                   (Debounced, Single-Press, Cooldown)
         │                                              │
         └──────────────────────┬───────────────────────┘
                                ▼
                         Input Manager
            [Continuous Key-Hold vs Single-Tap Rules]
                                │
                                ▼
                   Keyboard Controller Driver
                (pynput / pyautogui + Safety Guard)
                                │
                                ▼
              Existing Online Fighting Game (Browser)
```

---

## 🎮 Game Controls & Gesture Reference

### 1. Movement Controls (Left Hand) — Continuous Key Hold
Movement works like a virtual analog stick centered on a calibrated neutral anchor. Crossing the deadzone boundary keeps the corresponding key held down continuously. Returning to the deadzone automatically releases the key.

| Direction | Action | Trigger Condition | Key Behavior |
| :--- | :--- | :--- | :--- |
| **Move Left** | Move Left | Hand moves left past deadzone | **Hold `A`** |
| **Move Right** | Move Right | Hand moves right past deadzone | **Hold `D`** |
| **Jump / Up** | Jump | Hand moves up past deadzone | **Hold `W`** |
| **Crouch / Down** | Crouch | Hand moves down past deadzone | **Hold `S`** |
| **Neutral** | Idle / Stop | Hand rests inside deadzone box | **Release All** |

### 2. Combat Actions (Right Hand) — Single Key Tap
Combat attacks use geometric gesture classification with **temporal debouncing** and a **single-shot latch**. Holding a gesture will fire the mapped attack key **EXACTLY ONCE**. You must relax the gesture or switch gestures and wait for the cooldown before firing again.

| Gesture | Visual Description | Mapped Action | Key Pressed |
| :--- | :--- | :--- | :--- |
| **Open Palm** | All 5 fingers extended outwards | Attack / Action 1 | **Tap `Y`** |
| **Closed Fist** | All fingers curled tightly into palm | Attack / Action 2 | **Tap `H`** |
| **V-Sign (Peace)** | Index & Middle extended, Ring & Pinky curled | Attack / Action 3 | **Tap `G`** |
| **Thumbs Up** | Thumb pointing up, other 4 fingers curled | Attack / Action 4 | **Tap `J`** |

---

## 🚀 Quick Start & Installation (Windows)

No GPU, CUDA, Docker, or complex setup required. CV Fighter runs efficiently on CPU with standard Python.

### Prerequisites
- Windows 10 or 11 (also runs on macOS / Linux)
- Python 3.11 or Python 3.12
- Laptop webcam or USB webcam

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vidit0305/CV-Fighting-Project.git
   cd CV-Fighting-Project
   ```

2. **Create and Activate a Virtual Environment**:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
   *(On Linux/macOS: `source venv/bin/activate`)*

3. **Install Dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

4. **Verify Installation (Run Test Suite)**:
   ```cmd
   python test_system.py
   ```
   *All 10 unit tests should pass with `OK`.*

5. **Launch CV Fighter**:
   ```cmd
   python main.py
   ```

---

## 🕹️ How to Play with Your Browser Game

1. **Launch CV Fighter**:
   ```cmd
   python main.py
   ```
2. **Start in Test Mode**:
   - By default, CV Fighter launches in **TEST MODE** (safe simulation).
   - Position your hands in front of the webcam:
     - Your **left hand** will appear on the left side of the screen.
     - Your **right hand** will appear on the right side of the screen.
   - Verify that your movements and gestures are recognized on the OpenCV Heads-Up Display (HUD).
3. **Calibrate Neutral Anchor**:
   - Hold your left hand in a comfortable resting position.
   - Press **`C`** on your keyboard to calibrate the movement center.
4. **Open Your Existing Browser Fighting Game**:
   - In your browser (Chrome, Edge, Firefox), navigate to your fighting game.
5. **Switch to Live Game Control**:
   - Return to the CV Fighter window and press **`T`** to toggle into **LIVE GAME CONTROL MODE** (badge turns bright green).
6. **Focus the Browser Game Window (CRITICAL)**:
   - **Click inside your browser game window** to ensure it has active keyboard focus!
   - Because CV Fighter behaves like a physical keyboard sending OS keystrokes, your browser window must be active to receive `W`, `A`, `S`, `D`, `Y`, `H`, `G`, `J`.
7. **Play with Gestures**:
   - Move your left hand to run, jump, and crouch.
   - Flash your right hand gestures to punch, kick, and perform combos!

---

## ⌨️ Runtime Hotkeys & Safety Mechanisms

| Key | Function | Description |
| :--- | :--- | :--- |
| **`ESC`** or **`Q`** | **Emergency Stop & Exit** | Instantly releases all held keys, turns off webcam, and exits safely. |
| **`T`** | **Toggle Test / Live Mode** | Safely switches between test simulation and live OS keystrokes. |
| **`C`** | **Calibrate Anchor** | Calibrates the movement neutral center to your current hand position. |
| **`D`** | **Toggle Debug Overlay** | Shows landmark IDs, finger extension states, coordinates, and timers. |

### 🛡️ Fail-Safe Safety Guarantee
The controller tracks every key pressed in memory. If:
- The webcam disconnects
- MediaPipe loses hand tracking
- You press `ESC` or interrupt with `Ctrl+C`
- The application crashes or exits

The emergency cleanup handler (`KeyboardController.release_all`) triggers automatically via `atexit` and `try...finally` to ensure **no keys are ever left stuck down**.

---

## ⚙️ Configuration (`config.py`)

All parameters are easily customizable in `config.py`:

```python
# Webcam device index (0 = default laptop webcam)
CAMERA_INDEX = 0

# Video resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Virtual joystick deadzone thresholds (fraction of screen)
MOVEMENT_DEADZONE_X = 0.08
MOVEMENT_DEADZONE_Y = 0.08

# Debouncing: consecutive frames required to trigger attack
GESTURE_STABILITY_FRAMES = 4

# Attack cooldown in seconds
ATTACK_COOLDOWN_SEC = 0.35

# Key mappings
KEY_MAPPINGS = {
    "UP": "w",
    "LEFT": "a",
    "DOWN": "s",
    "RIGHT": "d",
    "ATTACK_1": "y",  # Open Palm
    "ATTACK_2": "h",  # Closed Fist
    "ATTACK_3": "g",  # V-Sign (Peace)
    "ATTACK_4": "j",  # Thumbs Up
}
```

---

## 📁 Project Structure

```
CV-Fighter/
├── main.py                     # Main application entry point and game loop
├── config.py                   # Central configuration, key mappings, and thresholds
├── requirements.txt            # Pinned dependencies (OpenCV, MediaPipe, pynput, etc.)
├── test_system.py              # Automated unit tests for gesture & input logic
├── .gitignore                  # Git ignore rules
├── README.md                   # Comprehensive user guide and documentation
│
├── vision/
│   ├── __init__.py
│   ├── hand_detector.py        # MediaPipe Hands wrapper, 21 landmarks, skeleton rendering
│   ├── gesture_recognizer.py   # Geometric gesture classification and debouncing filters
│   └── movement_detector.py    # Virtual D-pad, deadzone calculations, and EMA smoothing
│
├── controller/
│   ├── __init__.py
│   ├── keyboard_controller.py  # OS keyboard event generator (pynput/pyautogui) with safety release
│   └── input_manager.py        # Orchestrates key-hold, single-tap, cooldowns, and test mode
│
└── utils/
    ├── __init__.py
    ├── fps_counter.py          # High-precision rolling FPS tracker
    └── hud_display.py          # Cyberpunk gaming HUD overlay with visual feedback
```

---

## 🔧 Troubleshooting & FAQ

#### 1. "ERROR: Unable to open webcam at index 0"
- Ensure no other application (Zoom, Microsoft Teams, Discord, browser) is actively using the camera.
- If you have an external USB webcam, try running `python main.py --camera 1`.
- On Windows 10/11: Open **Windows Settings** $\rightarrow$ **Privacy & Security** $\rightarrow$ **Camera** $\rightarrow$ enable **"Allow desktop apps to access your camera"**.

#### 2. "The gestures are detected on screen, but my game isn't moving"
- Verify that **LIVE GAME CONTROL** is active (Press **`T`** to exit Test Mode).
- **Click into your browser window** so the browser game has active operating system focus. Background windows do not receive keyboard input.

#### 3. "Attacks trigger too easily or too slowly"
- In `config.py`, increase `GESTURE_STABILITY_FRAMES` (e.g. from `4` to `6`) for stricter debouncing.
- Adjust `ATTACK_COOLDOWN_SEC` (e.g. from `0.35` to `0.25`) for faster attack responsiveness.

#### 4. "Movement feels too sensitive"
- Press **`C`** while holding your hand naturally to recalibrate the center.
- Increase `MOVEMENT_DEADZONE_X` and `MOVEMENT_DEADZONE_Y` in `config.py` (e.g., from `0.08` to `0.12`).

---

## 📜 License
MIT License. Built for real-time computer vision gaming.
