"""Audio-only stock-ACC transitions for Sharan; never change control events."""
from cereal import car

from openpilot.common.realtime import DT_CTRL
from openpilot.sunnypilot.selfdrive.controls.lib.sharan_tuning import is_sharan

AudibleAlert = car.CarControl.HUDControl.AudibleAlert
CHIME_FRAMES = max(1, int(0.2 / DT_CTRL))


class StockAccChime:
  def __init__(self, CP):
    self.available = is_sharan(CP.carFingerprint) and CP.pcmCruise and not CP.openpilotLongitudinalControl and not CP.passive
    self._previous_enabled: bool | None = None
    self._sound = AudibleAlert.none
    self._until_frame = -1

  def update(self, CS, initialized: bool, frame: int) -> None:
    if not self.available or not initialized or not CS.canValid:
      self._previous_enabled = None
      self._sound = AudibleAlert.none
      self._until_frame = -1
      return

    enabled = bool(CS.cruiseState.enabled)
    previous = self._previous_enabled
    self._previous_enabled = enabled
    # First valid sample establishes a baseline; startup/reconnection must not chime.
    if previous is not None and enabled != previous:
      self._sound = AudibleAlert.engage if enabled else AudibleAlert.disengage
      self._until_frame = frame + CHIME_FRAMES

  def get_sound(self, native_sound, frame: int):
    # All native audible alerts win, including the normal engage/disengage chime.
    # Drop the fallback instead of replaying a stale transition after a warning.
    if native_sound != AudibleAlert.none:
      self._sound = AudibleAlert.none
      return native_sound
    if frame < self._until_frame:
      return self._sound
    self._sound = AudibleAlert.none
    return AudibleAlert.none
