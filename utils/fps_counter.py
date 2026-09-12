"""
High-precision FPS Counter for CV Fighter
Computes smoothed rolling-average frames per second.
"""

from collections import deque
import time


class FPSCounter:
    """Calculates smoothed frames per second using a moving time window."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: deque = deque(maxlen=window_size)
        self.last_time = time.perf_counter()
        self.current_fps: float = 0.0

    def update(self) -> float:
        """Records a new frame and updates the current FPS."""
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now

        if dt > 0:
            self.frame_times.append(dt)

        if len(self.frame_times) > 0:
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        else:
            self.current_fps = 0.0

        return self.current_fps

    def get_fps(self) -> float:
        """Returns the current smoothed FPS as float."""
        return self.current_fps

    def get_fps_str(self) -> str:
        """Returns the current FPS formatted as a string."""
        return f"{self.current_fps:.1f}"
