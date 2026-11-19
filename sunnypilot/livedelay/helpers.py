"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params
from openpilot.sunnypilot.selfdrive.controls.lib.sharan_tuning import is_sharan, sharan_lateral_delay


def get_lat_delay(params: Params, stock_lat_delay: float, car_fingerprint: str | None = None) -> float:
  # Apply the same Sharan source in model prediction and controller feedback.
  # None asks for a fixed override without confusing it with an equal live value.
  # Fixed Sharan control must not read Params in its 100 Hz update loop.
  fixed_delay = sharan_lateral_delay(car_fingerprint, None)
  if fixed_delay is not None:
    return fixed_delay

  if params.get_bool("LagdToggle"):
    return sharan_lateral_delay(car_fingerprint, float(params.get("LagdValueCache", return_default=True)))

  return stock_lat_delay


def get_controller_lat_delay(params: Params, stock_lat_delay: float, car_fingerprint: str) -> float:
  # Only Sharan changes source selection; preserve other controllers' native behavior.
  if is_sharan(car_fingerprint):
    return get_lat_delay(params, stock_lat_delay, car_fingerprint)
  return stock_lat_delay
