"""
HUD Display Module for CV Fighter
Draws a gaming-grade, cyberpunk-styled Heads-Up Display directly on the OpenCV canvas:
  - Live FPS and camera diagnostics
  - Mode badge (Test Mode vs Live Game Control)
  - Movement D-Pad / virtual joystick with deadzone boundaries and directional cues
  - Action / attack gesture recognizer cards with stability bars and trigger flashes
  - Comprehensive debug diagnostics overlay
"""

from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
import time

from config import COLORS, KEY_MAPPINGS, GESTURE_NAMES
from controller.input_manager import InputManager
from vision.gesture_recognizer import GestureState
from vision.hand_detector import HandData
from vision.movement_detector import MovementState


class HUDDisplay:
    """Renders all visual feedback, status badges, and diagnostics on the video frame."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_bold = cv2.FONT_HERSHEY_DUPLEX
        self.calibration_banner_time: float = 0.0

    @staticmethod
    def draw_transparent_rect(
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        alpha: float = 0.5,
    ) -> None:
        """Draws a semi-transparent filled rectangle on the frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def trigger_calibration_notice(self) -> None:
        """Triggers a temporary on-screen notice that anchor was calibrated."""
        self.calibration_banner_time = time.time()

    def render(
        self,
        frame: np.ndarray,
        movement_state: MovementState,
        gesture_state: GestureState,
        input_mgr: InputManager,
        fps_str: str,
        test_mode: bool,
        debug_mode: bool,
        movement_hand: Optional[HandData],
        attack_hand: Optional[HandData],
    ) -> np.ndarray:
        """Renders the complete HUD overlay onto the given frame."""
        h, w, _ = frame.shape

        # -------------------------------------------------------------
        # 1. TOP HEADER BAR
        # -------------------------------------------------------------
        self.draw_transparent_rect(frame, (0, 0), (w, 55), (15, 15, 20), alpha=0.75)
        cv2.line(frame, (0, 55), (w, 55), COLORS["PRIMARY"], 2)

        # Title
        cv2.putText(
            frame,
            "CV FIGHTER",
            (20, 36),
            self.font_bold,
            0.9,
            COLORS["PRIMARY"],
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "CONTROLLER",
            (200, 36),
            self.font,
            0.65,
            COLORS["WHITE"],
            1,
            cv2.LINE_AA,
        )

        # Camera status
        cv2.circle(frame, (380, 28), 6, COLORS["SUCCESS"], -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "CAMERA: ACTIVE",
            (395, 33),
            self.font,
            0.5,
            COLORS["NEUTRAL"],
            1,
            cv2.LINE_AA,
        )

        # FPS Display
        cv2.putText(
            frame,
            f"FPS: {fps_str}",
            (560, 33),
            self.font,
            0.55,
            COLORS["SECONDARY"],
            1,
            cv2.LINE_AA,
        )

        # Mode Badge (Center-Right)
        if test_mode:
            badge_text = "TEST MODE — INPUT DISABLED (Press T)"
            badge_color = (0, 220, 255)  # Bright Yellow/Amber
            bg_color = (10, 80, 100)
        else:
            badge_text = "LIVE GAME CONTROL ACTIVE (Press T)"
            badge_color = COLORS["SUCCESS"]
            bg_color = (10, 80, 20)

        badge_x = w - 460
        self.draw_transparent_rect(frame, (badge_x - 10, 10), (w - 20, 45), bg_color, alpha=0.85)
        cv2.rectangle(frame, (badge_x - 10, 10), (w - 20, 45), badge_color, 1)
        cv2.putText(
            frame,
            badge_text,
            (badge_x, 32),
            self.font,
            0.52,
            badge_color,
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 2. LEFT PANEL: MOVEMENT CONTROLS (LEFT HAND)
        # -------------------------------------------------------------
        panel_w = 260
        panel_y1 = 65
        panel_y2 = 230
        self.draw_transparent_rect(frame, (15, panel_y1), (15 + panel_w, panel_y2), (18, 18, 24), alpha=0.7)
        cv2.rectangle(frame, (15, panel_y1), (15 + panel_w, panel_y2), (60, 60, 80), 1)

        cv2.putText(
            frame,
            "LEFT HAND — MOVEMENT",
            (25, panel_y1 + 25),
            self.font_bold,
            0.52,
            COLORS["PRIMARY"],
            1,
            cv2.LINE_AA,
        )

        # Tracking status
        is_mov_tracked = movement_hand is not None
        status_text = "TRACKED" if is_mov_tracked else "SEARCHING..."
        status_color = COLORS["SUCCESS"] if is_mov_tracked else COLORS["WARNING"]
        cv2.putText(
            frame,
            f"Status: {status_text}",
            (25, panel_y1 + 50),
            self.font,
            0.48,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Direction and active key
        dir_text = movement_state.primary_direction
        cv2.putText(
            frame,
            f"Direction: {dir_text}",
            (25, panel_y1 + 80),
            self.font_bold,
            0.6,
            COLORS["WHITE"] if dir_text == "NEUTRAL" else COLORS["SECONDARY"],
            1,
            cv2.LINE_AA,
        )

        held_keys_str = " + ".join([k.upper() for k in sorted(input_mgr.held_movement_keys)]) or "NONE"
        cv2.putText(
            frame,
            f"Key Held: [{held_keys_str}]",
            (25, panel_y1 + 115),
            self.font,
            0.55,
            COLORS["ACTIVE_ZONE"] if held_keys_str != "NONE" else COLORS["NEUTRAL"],
            2 if held_keys_str != "NONE" else 1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Action: Continuous Hold",
            (25, panel_y1 + 145),
            self.font,
            0.42,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 3. MOVEMENT JOYSTICK / DEADZONE OVERLAY ON SCREEN
        # -------------------------------------------------------------
        # Pixel coordinates for neutral anchor and deadzones
        anchor_px_x = int(movement_state.anchor_pos[0] * w)
        anchor_px_y = int(movement_state.anchor_pos[1] * h)
        deadzone_px_w = int(0.08 * w)
        deadzone_px_h = int(0.08 * h)

        # Deadzone box
        dz_top_left = (anchor_px_x - deadzone_px_w, anchor_px_y - deadzone_px_h)
        dz_bottom_right = (anchor_px_x + deadzone_px_w, anchor_px_y + deadzone_px_h)
        cv2.rectangle(frame, dz_top_left, dz_bottom_right, (70, 70, 90), 1, cv2.LINE_AA)

        # Anchor center crosshair
        cv2.circle(frame, (anchor_px_x, anchor_px_y), 4, (120, 120, 140), -1, cv2.LINE_AA)
        cv2.line(frame, (anchor_px_x - 10, anchor_px_y), (anchor_px_x + 10, anchor_px_y), (100, 100, 120), 1)
        cv2.line(frame, (anchor_px_x, anchor_px_y - 10), (anchor_px_x, anchor_px_y + 10), (100, 100, 120), 1)

        # Directional labels around deadzone
        cv2.putText(frame, "W", (anchor_px_x - 7, dz_top_left[1] - 8), self.font_bold, 0.45, (160, 160, 160), 1)
        cv2.putText(frame, "S", (anchor_px_x - 6, dz_bottom_right[1] + 18), self.font_bold, 0.45, (160, 160, 160), 1)
        cv2.putText(frame, "A", (dz_top_left[0] - 22, anchor_px_y + 5), self.font_bold, 0.45, (160, 160, 160), 1)
        cv2.putText(frame, "D", (dz_bottom_right[0] + 10, anchor_px_y + 5), self.font_bold, 0.45, (160, 160, 160), 1)

        # If movement hand is tracked, draw vector line from anchor to current palm center
        if is_mov_tracked:
            curr_px_x = int(movement_state.current_pos[0] * w)
            curr_px_y = int(movement_state.current_pos[1] * h)
            in_neutral = len(movement_state.active_directions) == 0
            line_color = COLORS["PRIMARY"] if not in_neutral else (100, 180, 100)

            # Vector line
            cv2.line(frame, (anchor_px_x, anchor_px_y), (curr_px_x, curr_px_y), line_color, 2, cv2.LINE_AA)
            # Palm position marker
            cv2.circle(frame, (curr_px_x, curr_px_y), 8, line_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (curr_px_x, curr_px_y), 10, COLORS["WHITE"], 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 4. RIGHT PANEL: ACTION / ATTACK CONTROLS (RIGHT HAND)
        # -------------------------------------------------------------
        action_x1 = w - panel_w - 15
        self.draw_transparent_rect(frame, (action_x1, panel_y1), (w - 15, panel_y2 + 20), (18, 18, 24), alpha=0.7)
        cv2.rectangle(frame, (action_x1, panel_y1), (w - 15, panel_y2 + 20), (60, 60, 80), 1)

        cv2.putText(
            frame,
            "RIGHT HAND — ACTIONS",
            (action_x1 + 10, panel_y1 + 25),
            self.font_bold,
            0.52,
            COLORS["SECONDARY"],
            1,
            cv2.LINE_AA,
        )

        is_act_tracked = attack_hand is not None
        act_status_text = "TRACKED" if is_act_tracked else "SEARCHING..."
        act_status_color = COLORS["SUCCESS"] if is_act_tracked else COLORS["WARNING"]
        cv2.putText(
            frame,
            f"Status: {act_status_text}",
            (action_x1 + 10, panel_y1 + 50),
            self.font,
            0.48,
            act_status_color,
            1,
            cv2.LINE_AA,
        )

        # Detected Gesture
        gest_label = GESTURE_NAMES.get(gesture_state.confirmed_gesture, gesture_state.confirmed_gesture)
        cv2.putText(
            frame,
            f"Gesture: {gest_label}",
            (action_x1 + 10, panel_y1 + 80),
            self.font_bold,
            0.55,
            COLORS["WHITE"] if gest_label == "Neutral / Idle" else COLORS["PRIMARY"],
            1,
            cv2.LINE_AA,
        )

        # Mapped Action & Key
        action_name = input_mgr.gesture_to_action.get(gesture_state.confirmed_gesture, "NONE")
        action_key = KEY_MAPPINGS.get(action_name, "-").upper()
        action_text = f"Action: {action_key} ({action_name})" if action_name != "NONE" else "Action: None"
        cv2.putText(
            frame,
            action_text,
            (action_x1 + 10, panel_y1 + 115),
            self.font_bold,
            0.55,
            COLORS["SECONDARY"] if action_name != "NONE" else COLORS["NEUTRAL"],
            1,
            cv2.LINE_AA,
        )

        # Stability progress bar
        bar_x = action_x1 + 10
        bar_y = panel_y1 + 130
        bar_w = panel_w - 20
        bar_h = 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 50), -1)
        stability_fill = int(bar_w * (gesture_state.stability_count / max(1, 4)))
        fill_color = COLORS["SUCCESS"] if gesture_state.is_stable else COLORS["WARNING"]
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + stability_fill, bar_y + bar_h), fill_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 100), 1)

        cv2.putText(
            frame,
            f"Stability: {gesture_state.stability_count}/4",
            (action_x1 + 10, panel_y1 + 155),
            self.font,
            0.42,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        # Cooldown indicator
        cooldown_elapsed = time.time() - input_mgr.last_attack_time
        on_cooldown = cooldown_elapsed < input_mgr.attack_cooldown_sec
        cd_text = f"Cooldown: {max(0.0, input_mgr.attack_cooldown_sec - cooldown_elapsed):.2f}s" if on_cooldown else "Cooldown: READY"
        cd_color = COLORS["WARNING"] if on_cooldown else COLORS["SUCCESS"]
        cv2.putText(
            frame,
            cd_text,
            (action_x1 + 130, panel_y1 + 155),
            self.font,
            0.42,
            cd_color,
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 5. ACTION TRIGGER FLASH BANNER (WHEN KEY FIRES)
        # -------------------------------------------------------------
        last_action, last_key, is_fresh = input_mgr.get_action_feedback()
        if is_fresh and last_key:
            flash_y = panel_y2 + 40
            self.draw_transparent_rect(frame, (action_x1, flash_y), (w - 15, flash_y + 40), (20, 100, 20), alpha=0.85)
            cv2.rectangle(frame, (action_x1, flash_y), (w - 15, flash_y + 40), COLORS["SUCCESS"], 2)
            cv2.putText(
                frame,
                f">> FIRED KEY: [{last_key.upper()}] <<",
                (action_x1 + 25, flash_y + 26),
                self.font_bold,
                0.58,
                COLORS["WHITE"],
                2,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 6. CALIBRATION NOTICE (WHEN USER PRESSES 'C')
        # -------------------------------------------------------------
        if (time.time() - self.calibration_banner_time) < 2.0:
            cal_w = 400
            cal_x = (w - cal_w) // 2
            self.draw_transparent_rect(frame, (cal_x, 70), (cal_x + cal_w, 120), (30, 80, 30), alpha=0.9)
            cv2.rectangle(frame, (cal_x, 70), (cal_x + cal_w, 120), COLORS["SUCCESS"], 2)
            cv2.putText(
                frame,
                "NEUTRAL ANCHOR CALIBRATED!",
                (cal_x + 30, 102),
                self.font_bold,
                0.62,
                COLORS["WHITE"],
                2,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 7. DEBUG OVERLAY (IF DEBUG_MODE IS ON)
        # -------------------------------------------------------------
        if debug_mode:
            self._render_debug_info(frame, movement_state, gesture_state, input_mgr, movement_hand, attack_hand)

        # -------------------------------------------------------------
        # 8. BOTTOM CONTROLS FOOTER
        # -------------------------------------------------------------
        footer_y = h - 35
        self.draw_transparent_rect(frame, (0, footer_y), (w, h), (15, 15, 20), alpha=0.8)
        cv2.line(frame, (0, footer_y), (w, footer_y), (40, 40, 50), 1)

        controls_text = "[ESC] Emergency Stop & Exit   |   [T] Toggle Test/Live Mode   |   [C] Calibrate Anchor   |   [D] Toggle Debug"
        cv2.putText(
            frame,
            controls_text,
            (25, h - 12),
            self.font,
            0.45,
            COLORS["NEUTRAL"],
            1,
            cv2.LINE_AA,
        )

        return frame

    def _render_debug_info(
        self,
        frame: np.ndarray,
        movement_state: MovementState,
        gesture_state: GestureState,
        input_mgr: InputManager,
        movement_hand: Optional[HandData],
        attack_hand: Optional[HandData],
    ) -> None:
        """Renders comprehensive diagnostics for troubleshooting."""
        h, w, _ = frame.shape
        debug_w = 520
        debug_h = 135
        x1 = 15
        y1 = h - 35 - debug_h - 10
        x2 = x1 + debug_w
        y2 = y1 + debug_h

        self.draw_transparent_rect(frame, (x1, y1), (x2, y2), (10, 10, 15), alpha=0.85)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["SECONDARY"], 1)

        cv2.putText(frame, "DEBUG METRICS OVERLAY", (x1 + 10, y1 + 20), self.font_bold, 0.48, COLORS["SECONDARY"], 1)

        # Finger states summary
        f = gesture_state.finger_states
        fingers_str = f"Thumb:{int(f['thumb'])} | Idx:{int(f['index'])} | Mid:{int(f['middle'])} | Rng:{int(f['ring'])} | Pnk:{int(f['pinky'])}"
        cv2.putText(frame, f"Fingers: {fingers_str}", (x1 + 10, y1 + 45), self.font, 0.42, COLORS["WHITE"], 1)

        # Coordinates
        curr_pos = movement_state.current_pos
        offset = movement_state.offset
        cv2.putText(
            frame,
            f"Mov Anchor: ({movement_state.anchor_pos[0]:.2f}, {movement_state.anchor_pos[1]:.2f}) | Curr: ({curr_pos[0]:.2f}, {curr_pos[1]:.2f}) | Offset: ({offset[0]:+.2f}, {offset[1]:+.2f})",
            (x1 + 10, y1 + 70),
            self.font,
            0.40,
            COLORS["NEUTRAL"],
            1,
        )

        # Gesture confidence & Raw
        cv2.putText(
            frame,
            f"Raw Gest: {gesture_state.raw_gesture} ({gesture_state.confidence*100:.0f}%) | Confirmed: {gesture_state.confirmed_gesture} | Latched: {input_mgr.action_latched}",
            (x1 + 10, y1 + 95),
            self.font,
            0.40,
            COLORS["NEUTRAL"],
            1,
        )

        # Driver & All Held Keys
        held_keys = list(input_mgr.keyboard.get_held_keys())
        cv2.putText(
            frame,
            f"Backend: {input_mgr.keyboard.backend_name} | OS Held Keys: {held_keys}",
            (x1 + 10, y1 + 120),
            self.font,
            0.40,
            COLORS["SUCCESS"] if held_keys else COLORS["NEUTRAL"],
            1,
        )
