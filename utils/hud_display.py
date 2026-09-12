"""
HUD Display Module for CV Fighter
Clean, minimalist, gaming-grade Heads-Up Display:
- Minimal floating top pill for mode and FPS (no giant opaque header boxes)
- Non-intrusive directional pills that only appear when moving
- Sleek action chips that only appear when combat gestures are detected
- Edge-to-edge fullscreen clean canvas
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
    """Renders a sleek, ultra-clean, minimalist gaming HUD directly on the frame."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_bold = cv2.FONT_HERSHEY_DUPLEX
        self.calibration_banner_time: float = 0.0

    def trigger_calibration_notice(self) -> None:
        """Triggers a temporary on-screen notice that anchor was calibrated."""
        self.calibration_banner_time = time.time()

    @staticmethod
    def draw_glass_pill(
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        bg_color: Tuple[int, int, int] = (15, 15, 20),
        border_color: Tuple[int, int, int] = (60, 60, 80),
        alpha: float = 0.65,
        radius: int = 10,
    ) -> None:
        """Draws a sleek translucent rounded pill."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)

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
        """Renders minimal, non-obtrusive HUD elements onto the frame."""
        h, w, _ = frame.shape

        # -------------------------------------------------------------
        # 1. FLOATING TOP STATUS PILL (COMPACT, SLEEK, UNCLUTTERED)
        # -------------------------------------------------------------
        pill_w = 440
        pill_h = 36
        pill_x = (w - pill_w) // 2
        pill_y = 15

        if test_mode:
            status_text = "TEST MODE — (Press T for Game Control)"
            dot_color = (0, 215, 255)  # Glowing Amber
            pill_border = (0, 180, 220)
            pill_bg = (12, 28, 38)
        else:
            status_text = "LIVE GAME CONTROLLER ACTIVE"
            dot_color = (60, 240, 80)   # Vivid Green
            pill_border = (50, 180, 70)
            pill_bg = (12, 34, 18)

        self.draw_glass_pill(frame, pill_x, pill_y, pill_w, pill_h, bg_color=pill_bg, border_color=pill_border, alpha=0.75)

        # Status Glowing Dot
        cv2.circle(frame, (pill_x + 18, pill_y + 18), 5, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (pill_x + 18, pill_y + 18), 7, (255, 255, 255), 1, cv2.LINE_AA)

        # Status Label
        cv2.putText(
            frame,
            status_text,
            (pill_x + 32, pill_y + 24),
            self.font_bold,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # FPS Tag on Right of Pill
        cv2.putText(
            frame,
            f"{fps_str} FPS",
            (pill_x + pill_w - 65, pill_y + 24),
            self.font,
            0.42,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 2. MOVEMENT JOYSTICK / DEADZONE (MINIMALIST & CLEAN)
        # -------------------------------------------------------------
        anchor_px_x = int(movement_state.anchor_pos[0] * w)
        anchor_px_y = int(movement_state.anchor_pos[1] * h)
        deadzone_px_w = int(0.08 * w)
        deadzone_px_h = int(0.08 * h)

        # Subtle thin deadzone reticle
        dz_color = (80, 80, 100) if not movement_state.active_directions else COLORS["PRIMARY"]
        cv2.circle(frame, (anchor_px_x, anchor_px_y), 4, dz_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (anchor_px_x, anchor_px_y), deadzone_px_w, (60, 60, 75), 1, cv2.LINE_AA)

        # Subtle small direction ticks
        cv2.line(frame, (anchor_px_x - deadzone_px_w, anchor_px_y), (anchor_px_x - deadzone_px_w + 8, anchor_px_y), (100, 100, 120), 1)
        cv2.line(frame, (anchor_px_x + deadzone_px_w - 8, anchor_px_y), (anchor_px_x + deadzone_px_w, anchor_px_y), (100, 100, 120), 1)
        cv2.line(frame, (anchor_px_x, anchor_px_y - deadzone_px_h), (anchor_px_x, anchor_px_y - deadzone_px_h + 8), (100, 100, 120), 1)
        cv2.line(frame, (anchor_px_x, anchor_px_y + deadzone_px_h - 8), (anchor_px_x, anchor_px_y + deadzone_px_h), (100, 100, 120), 1)

        # If movement hand is tracked, draw vector line to palm
        if movement_hand is not None:
            curr_px_x = int(movement_state.current_pos[0] * w)
            curr_px_y = int(movement_state.current_pos[1] * h)
            is_moving = len(movement_state.active_directions) > 0
            line_color = COLORS["PRIMARY"] if is_moving else (120, 180, 120)

            cv2.line(frame, (anchor_px_x, anchor_px_y), (curr_px_x, curr_px_y), line_color, 2, cv2.LINE_AA)
            cv2.circle(frame, (curr_px_x, curr_px_y), 6, line_color, -1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. DYNAMIC DIRECTIONAL PILL (APPEARS AT BOTTOM LEFT ONLY WHEN MOVING)
        # -------------------------------------------------------------
        if movement_state.active_directions:
            dir_str = " + ".join(sorted(movement_state.active_directions))
            keys_str = " + ".join([k.upper() for k in sorted(input_mgr.held_movement_keys)])

            arrow_map = {"LEFT": "◄", "RIGHT": "►", "UP": "▲", "DOWN": "▼"}
            arrow_icon = " ".join([arrow_map.get(d, "") for d in sorted(movement_state.active_directions)])

            mov_pill_w = 260
            mov_pill_h = 42
            mov_pill_x = 35
            mov_pill_y = h - 100

            self.draw_glass_pill(frame, mov_pill_x, mov_pill_y, mov_pill_w, mov_pill_h,
                                 bg_color=(15, 30, 45), border_color=COLORS["PRIMARY"], alpha=0.85)

            cv2.putText(
                frame,
                f"{arrow_icon} HOLD [{keys_str}] {dir_str}",
                (mov_pill_x + 18, mov_pill_y + 27),
                self.font_bold,
                0.55,
                COLORS["PRIMARY"],
                2,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 4. DYNAMIC ACTION PILL (APPEARS AT BOTTOM RIGHT ONLY WHEN GESTURE IS ACTIVE)
        # -------------------------------------------------------------
        confirmed_gest = gesture_state.confirmed_gesture
        raw_gest = gesture_state.raw_gesture

        # Show action feedback if an attack gesture is active or debouncing
        display_gest = confirmed_gest if confirmed_gest != "NONE" else raw_gest

        if display_gest in input_mgr.gesture_to_action:
            action_name = input_mgr.gesture_to_action[display_gest]
            mapped_key = KEY_MAPPINGS.get(action_name, "?").upper()
            gest_label = GESTURE_NAMES.get(display_gest, display_gest).upper()

            act_pill_w = 280
            act_pill_h = 46
            act_pill_x = w - act_pill_w - 35
            act_pill_y = h - 104

            # Pill background
            border_c = COLORS["SUCCESS"] if gesture_state.is_stable else COLORS["SECONDARY"]
            self.draw_glass_pill(frame, act_pill_x, act_pill_y, act_pill_w, act_pill_h,
                                 bg_color=(30, 25, 15), border_color=border_c, alpha=0.85)

            # Action Text
            cv2.putText(
                frame,
                f"[{mapped_key}]  {gest_label}",
                (act_pill_x + 16, act_pill_y + 26),
                self.font_bold,
                0.55,
                COLORS["WHITE"],
                1,
                cv2.LINE_AA,
            )

            # Stability Progress Bar inside the pill
            bar_x = act_pill_x + 16
            bar_y = act_pill_y + 34
            bar_w = act_pill_w - 32
            bar_h = 4
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
            fill_w = int(bar_w * (gesture_state.stability_count / 4))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), border_c, -1)

        # -------------------------------------------------------------
        # 5. ACTION TRIGGER FLASH BANNER (WHEN KEY FIRES)
        # -------------------------------------------------------------
        last_action, last_key, is_fresh = input_mgr.get_action_feedback()
        if is_fresh and last_key:
            flash_w = 260
            flash_h = 38
            flash_x = w - flash_w - 35
            flash_y = h - 150
            self.draw_glass_pill(frame, flash_x, flash_y, flash_w, flash_h,
                                 bg_color=(10, 60, 20), border_color=COLORS["SUCCESS"], alpha=0.9)
            cv2.putText(
                frame,
                f"⚡ TRIGGERED [{last_key.upper()}]",
                (flash_x + 24, flash_y + 25),
                self.font_bold,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 6. CALIBRATION NOTICE (WHEN USER PRESSES 'C')
        # -------------------------------------------------------------
        if (time.time() - self.calibration_banner_time) < 1.5:
            cal_w = 340
            cal_h = 42
            cal_x = (w - cal_w) // 2
            cal_y = 65
            self.draw_glass_pill(frame, cal_x, cal_y, cal_w, cal_h,
                                 bg_color=(15, 50, 20), border_color=COLORS["SUCCESS"], alpha=0.9)
            cv2.putText(
                frame,
                "✓ ANCHOR CALIBRATED",
                (cal_x + 40, cal_y + 27),
                self.font_bold,
                0.58,
                COLORS["WHITE"],
                2,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 7. DEBUG OVERLAY (ONLY WHEN USER TOGGLES 'D')
        # -------------------------------------------------------------
        if debug_mode:
            self._render_minimal_debug(frame, movement_state, gesture_state, input_mgr)

        # -------------------------------------------------------------
        # 8. SUBTLE BOTTOM CONTROL HINTS (ONE LOW-PROFILE LINE)
        # -------------------------------------------------------------
        hint_text = "[ESC] Exit   •   [T] Mode   •   [C] Calibrate   •   [F] Fullscreen   •   [D] Debug"
        cv2.putText(
            frame,
            hint_text,
            (w // 2 - 250, h - 16),
            self.font,
            0.42,
            (140, 140, 150),
            1,
            cv2.LINE_AA,
        )

        return frame

    def _render_minimal_debug(
        self,
        frame: np.ndarray,
        movement_state: MovementState,
        gesture_state: GestureState,
        input_mgr: InputManager,
    ) -> None:
        """Compact diagnostics overlay."""
        db_w = 400
        db_h = 80
        x = 25
        y = 65
        self.draw_glass_pill(frame, x, y, db_w, db_h, bg_color=(10, 10, 15), border_color=COLORS["SECONDARY"], alpha=0.85)

        f = gesture_state.finger_states
        f_str = f"Fingers: T:{int(f['thumb'])} I:{int(f['index'])} M:{int(f['middle'])} R:{int(f['ring'])} P:{int(f['pinky'])}"
        cv2.putText(frame, f_str, (x + 12, y + 24), self.font, 0.40, (240, 240, 240), 1, cv2.LINE_AA)

        offset = movement_state.offset
        pos_str = f"Offset: ({offset[0]:+.2f}, {offset[1]:+.2f}) | Primary: {movement_state.primary_direction}"
        cv2.putText(frame, pos_str, (x + 12, y + 46), self.font, 0.40, COLORS["NEUTRAL"], 1, cv2.LINE_AA)

        held_keys = list(input_mgr.keyboard.get_held_keys())
        cv2.putText(frame, f"OS Keys Held: {held_keys}", (x + 12, y + 68), self.font, 0.40, COLORS["SUCCESS"], 1, cv2.LINE_AA)
