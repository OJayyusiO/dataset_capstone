"""
Traffic light state plumbing.

Provides a single source of truth for "what colour is the light on frame N?"
across three input modes:

  1. LIVE CARLA   - wrap a carla.TrafficLight actor; state is read on demand
  2. RECORDED     - read a light_states.csv (frame,state) saved during recording
  3. REAL VIDEO   - read a manual schedule from analytics_config.yaml

All modes expose the same interface: state_at(frame_idx) -> 'red'|'yellow'|'green'|'unknown'

The project has explicit permission to read traffic light state directly from
the CARLA simulator (it is treated as ground-truth signal infrastructure, not
something to be inferred from camera vision).
"""

import csv
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


VALID_STATES = ('red', 'yellow', 'green', 'unknown')

# BGR colours for drawing a light-state indicator
STATE_COLORS = {
    'red': (0, 0, 255),
    'yellow': (0, 220, 220),
    'green': (0, 200, 0),
    'unknown': (150, 150, 150),
}


def draw_light_indicator(frame, state, top_right=True):
    """Draw a coloured circle + label showing the current light state.

    Placed in the top-right corner by default.
    """
    if cv2 is None:
        return
    h, w = frame.shape[:2]
    color = STATE_COLORS.get(state, STATE_COLORS['unknown'])
    label = f"LIGHT: {state.upper()}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    if top_right:
        x = w - tw - 60
        y = 30
    else:
        x = 20
        y = 30
    # dark background strip
    cv2.rectangle(frame, (x - 10, y - 22), (x + tw + 40, y + 10), (0, 0, 0), -1)
    # colored circle
    cv2.circle(frame, (x + tw + 22, y - 6), 10, color, -1)
    cv2.circle(frame, (x + tw + 22, y - 6), 10, (255, 255, 255), 1)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def carla_state_to_str(carla_state):
    """Convert a carla.TrafficLightState enum to our lowercase string."""
    name = str(carla_state).split('.')[-1].lower()  # e.g. "TrafficLightState.Red" -> "red"
    if name in VALID_STATES:
        return name
    return 'unknown'


class LightStateProvider:
    """Unified per-frame traffic light state lookup."""

    def __init__(self, mode, carla_light=None, schedule=None):
        self.mode = mode                # 'carla' | 'recorded' | 'schedule' | 'none'
        self._carla_light = carla_light
        # schedule: sorted list of (frame_idx, state) tuples
        self._schedule = sorted(schedule, key=lambda x: x[0]) if schedule else None

    # --- constructors -----------------------------------------------------

    @classmethod
    def from_carla(cls, carla_light):
        """Live mode: read the state from a CARLA traffic light actor on demand."""
        if carla_light is None:
            return cls(mode='none')
        return cls(mode='carla', carla_light=carla_light)

    @classmethod
    def from_csv(cls, csv_path):
        """Recorded mode: read a light_states.csv with columns frame,state."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return cls(mode='none')
        schedule = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame = int(row['frame'])
                except (KeyError, ValueError):
                    continue
                state = (row.get('state') or 'unknown').strip().lower()
                if state not in VALID_STATES:
                    state = 'unknown'
                schedule.append((frame, state))
        if not schedule:
            return cls(mode='none')
        return cls(mode='recorded', schedule=schedule)

    @classmethod
    def from_schedule(cls, schedule_entries):
        """Real-video mode: schedule_entries is a list of dicts {frame, state}.

        The active state for a frame is the most recent entry at or before it.
        """
        if not schedule_entries:
            return cls(mode='none')
        schedule = []
        for entry in schedule_entries:
            frame = int(entry.get('frame', 0))
            state = str(entry.get('state', 'unknown')).strip().lower()
            if state not in VALID_STATES:
                state = 'unknown'
            schedule.append((frame, state))
        if not schedule:
            return cls(mode='none')
        return cls(mode='schedule', schedule=schedule)

    # --- lookup -----------------------------------------------------------

    def state_at(self, frame_idx):
        """Return the light state for the given frame index."""
        if self.mode == 'carla':
            try:
                return carla_state_to_str(self._carla_light.get_state())
            except Exception:
                return 'unknown'

        if self.mode in ('recorded', 'schedule') and self._schedule:
            # Find the most recent schedule entry at or before frame_idx.
            current = 'unknown'
            for frame, state in self._schedule:
                if frame <= frame_idx:
                    current = state
                else:
                    break
            return current

        return 'unknown'

    @property
    def available(self):
        return self.mode != 'none'
