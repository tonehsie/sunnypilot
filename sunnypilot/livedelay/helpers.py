"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params
from openpilot.sunnypilot.selfdrive.controls.lib.sharan_tuning import sharan_lateral_delay


def get_lat_delay(params: Params, stock_lat_delay: float, car_fingerprint: str | None = None) -> float:
  # Apply the same Sharan source in model prediction and controller feedback.
  fixed_delay = sharan_lateral_delay(car_fingerprint, stock_lat_delay)
  if fixed_delay != stock_lat_delay:
    return fixed_delay

  if params.get_bool("LagdToggle"):
    return sharan_lateral_delay(car_fingerprint, float(params.get("LagdValueCache", return_default=True)))

  return stock_lat_delay
