"""
Keyboard Controller Module for CV Fighter
Generates real OS keyboard events (Key Down, Key Up, Key Tap)
with strict state tracking and fail-safe key release on exit.
"""

import atexit
import logging
import signal
import sys
import threading
import time
from typing import Set, Optional

logger = logging.getLogger(__name__)


class KeyboardController:
    """
    Low-level keyboard controller wrapper supporting pynput and pyautogui.
    Maintains an exact set of currently held keys and guarantees they are all
    released upon program exit or error.
    """

    def __init__(self, backend: str = "pynput"):
        self.backend_name = backend
        self._held_keys: Set[str] = set()
        self._lock = threading.Lock()
        self._driver: Optional[object] = None
        self._use_pynput = False

        self._init_driver()

        # Register fail-safe cleanup with Python's atexit
        atexit.register(self.release_all)

        # Handle SIGINT and SIGTERM gracefully
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, AttributeError):
            # Signal handling might be restricted in some non-main thread environments
            pass

    def _init_driver(self) -> None:
        """Initialize the requested or available keyboard driver."""
        if self.backend_name == "pynput":
            try:
                from pynput.keyboard import Controller as PynputController
                self._driver = PynputController()
                self._use_pynput = True
                logger.info("KeyboardController: Initialized using 'pynput' driver.")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize pynput driver ({e}). Falling back to pyautogui.")

        # Fallback to pyautogui
        try:
            import pyautogui
            pyautogui.PAUSE = 0.001  # Minimize delay
            pyautogui.FAILSAFE = False  # Prevent corner mouse crash during game control
            self._driver = pyautogui
            self._use_pynput = False
            logger.info("KeyboardController: Initialized using 'pyautogui' driver.")
        except Exception as e:
            logger.error(f"Failed to initialize pyautogui driver ({e}). Running in dummy mode.")
            self._driver = None

    def _signal_handler(self, sig, frame) -> None:
        """Emergency signal handler to release keys before exiting."""
        logger.warning(f"Received signal {sig}. Releasing all held keys and exiting.")
        self.release_all()
        sys.exit(0)

    def press(self, key: str) -> bool:
        """
        Sends a KeyDown event to the operating system.
        Returns True if newly pressed, False if already held.
        """
        key = key.lower()
        with self._lock:
            if key in self._held_keys:
                return False  # Already pressed

            try:
                if self._use_pynput and self._driver:
                    self._driver.press(key)
                elif self._driver:
                    self._driver.keyDown(key)
                self._held_keys.add(key)
                return True
            except Exception as e:
                logger.error(f"Error pressing key '{key}': {e}")
                return False

    def release(self, key: str) -> bool:
        """
        Sends a KeyUp event to the operating system.
        Returns True if released, False if wasn't held.
        """
        key = key.lower()
        with self._lock:
            if key not in self._held_keys:
                return False

            try:
                if self._use_pynput and self._driver:
                    self._driver.release(key)
                elif self._driver:
                    self._driver.keyUp(key)
            except Exception as e:
                logger.error(f"Error releasing key '{key}': {e}")
            finally:
                self._held_keys.discard(key)
            return True

    def tap(self, key: str, duration: float = 0.05) -> None:
        """
        Presses a key down, waits for `duration` seconds, and releases it.
        Runs in a lightweight background thread to prevent blocking the vision loop.
        """
        key = key.lower()

        def _tap_worker():
            self.press(key)
            time.sleep(duration)
            self.release(key)

        threading.Thread(target=_tap_worker, daemon=True).start()

    def release_all(self) -> None:
        """
        Safety function: releases all keys currently tracked as held.
        Guarantees keys don't get stuck down when the game or script closes.
        """
        with self._lock:
            keys_to_release = list(self._held_keys)
            for key in keys_to_release:
                try:
                    if self._use_pynput and self._driver:
                        self._driver.release(key)
                    elif self._driver:
                        self._driver.keyUp(key)
                except Exception as e:
                    logger.error(f"Error releasing key '{key}' during cleanup: {e}")
            self._held_keys.clear()

    def is_pressed(self, key: str) -> bool:
        """Returns True if the given key is currently pressed."""
        return key.lower() in self._held_keys

    def get_held_keys(self) -> Set[str]:
        """Returns a snapshot of all currently held keys."""
        with self._lock:
            return set(self._held_keys)
