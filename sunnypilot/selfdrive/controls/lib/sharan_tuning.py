"""Sharan-only settings shared by torque V0/V1, controls and model prediction.

These switches select consumed values; they do not stop the estimators.
No PID gains, CAN rate limits or Panda safety limits are changed here.
"""

SHARAN_FINGERPRINT = "VOLKSWAGEN_SHARAN_MK2"

# True: use the estimator. False: consume the corresponding fixed value.
USE_LIVE_STEER_RATIO = True
FIXED_STEER_RATIO = 15.6
USE_LIVE_STIFFNESS_FACTOR = True
FIXED_STIFFNESS_FACTOR = 1.0
USE_LIVE_ANGLE_OFFSET = True
FIXED_ANGLE_OFFSET_DEG = 0.0
USE_LIVE_ROLL = True
FIXED_ROLL = 0.0  # radians

USE_LIVE_TORQUE_PARAMS = False
FIXED_LAT_ACCEL_OFFSET = 0.0
FIXED_FRICTION = 0.08
SHARAN_FACTOR_TRANSITION_KPH = 35.0
SHARAN_FACTOR_STOPPED = 5.0
SHARAN_FACTOR_CRUISE = 1.8

USE_LIVE_DELAY = False
FIXED_LATERAL_DELAY = 0.2  # seconds, before each model's original smoothing term


def is_sharan(car_fingerprint):
  return car_fingerprint == SHARAN_FINGERPRINT


def sharan_lat_accel_factor(v_ego):
  speed_fraction = min(max(v_ego * 3.6 / SHARAN_FACTOR_TRANSITION_KPH, 0.0), 1.0)
  smooth_fraction = speed_fraction * speed_fraction * (3.0 - 2.0 * speed_fraction)
  return SHARAN_FACTOR_CRUISE + (SHARAN_FACTOR_STOPPED - SHARAN_FACTOR_CRUISE) * (1.0 - smooth_fraction)


def apply_sharan_torque_params(torque_params, v_ego):
  if USE_LIVE_TORQUE_PARAMS:
    return False

  changed = False
  for field, value in (("latAccelFactor", sharan_lat_accel_factor(v_ego)),
                       ("latAccelOffset", FIXED_LAT_ACCEL_OFFSET), ("friction", FIXED_FRICTION)):
    if getattr(torque_params, field) != value:
      setattr(torque_params, field, value)
      changed = True
  return changed


def sharan_live_parameters(car_fingerprint, live_params):
  if not is_sharan(car_fingerprint):
    return live_params

  fixed_fields = (("steerRatio", USE_LIVE_STEER_RATIO, FIXED_STEER_RATIO),
                  ("stiffnessFactor", USE_LIVE_STIFFNESS_FACTOR, FIXED_STIFFNESS_FACTOR),
                  ("angleOffsetDeg", USE_LIVE_ANGLE_OFFSET, FIXED_ANGLE_OFFSET_DEG),
                  ("roll", USE_LIVE_ROLL, FIXED_ROLL))
  if all(use_live for _, use_live, _ in fixed_fields):
    return live_params

  params = live_params.as_builder()
  for field, use_live, value in fixed_fields:
    if not use_live:
      setattr(params, field, value)
  return params


def sharan_lateral_delay(car_fingerprint, original_delay):
  if is_sharan(car_fingerprint) and not USE_LIVE_DELAY:
    return FIXED_LATERAL_DELAY
  return original_delay
