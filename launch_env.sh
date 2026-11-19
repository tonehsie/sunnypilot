#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# models get lower priority than ui
# - ui is ~5ms
# - modeld is 20ms
# - DM is 10ms
# in order to run ui at 60fps (16.67ms), we need to allow
# it to preempt the model workloads. we have enough
# headroom for this until ui is moved to the CPU.
export QCOM_PRIORITY=12

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="18.4"
fi

export STAGING_ROOT="/data/safe_staging"

# Mount the user's external ext4 recording partition before loggerd starts.
# Prefer /dev/disk/by-uuid/<UUID> here once its UUID is known; sdX names can move.
# Override EXTERNAL_RECORDING_DEVICE to select a different block-device path.
_sp_mount_external_recording() {
  local target="/data/media/0/realdata"
  local device="${EXTERNAL_RECORDING_DEVICE:-/dev/sdg1}"
  local cmd mounted_source mounted_type

  # launch_env.sh is also sourced by desktop tools; never mount on those hosts.
  [ -f /AGNOS ] || return 0
  for cmd in timeout sudo readlink mountpoint findmnt; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "[external-recording] Missing $cmd; leaving recording storage unchanged." >&2
      return 0
    fi
  done
  if [ ! -b "$device" ]; then
    echo "[external-recording] $device is unavailable; leaving recording storage unchanged." >&2
    return 0
  fi
  if ! device="$(readlink -f -- "$device")" || [ -z "$device" ]; then
    echo "[external-recording] Cannot resolve recording device; skipping mount." >&2
    return 0
  fi
  if [ -L "$target" ]; then
    echo "[external-recording] Refusing to mount over a symlink: $target" >&2
    return 0
  fi

  if ! mountpoint -q -- "$target"; then
    if findmnt -rn -S "$device" >/dev/null 2>&1; then
      echo "[external-recording] $device is already mounted elsewhere; leaving it unchanged." >&2
      return 0
    fi
    if ! timeout -k 2s 5s sudo -n mkdir -p -- "$target"; then
      echo "[external-recording] Cannot create $target; skipping external storage." >&2
      return 0
    fi
    if ! timeout -k 2s 10s sudo -n mount -t ext4 -o rw,nosuid,nodev -- "$device" "$target"; then
      echo "[external-recording] Mount failed/timed out; check storage before driving." >&2
      return 0
    fi
  fi

  # Never change permissions on a different filesystem already mounted there.
  if ! mounted_source="$(findmnt -rn -M "$target" -o SOURCE)" ||
     ! mounted_type="$(findmnt -rn -M "$target" -o FSTYPE)" ||
     ! mounted_source="$(readlink -f -- "$mounted_source")" ||
     [ "$mounted_source" != "$device" ] || [ "$mounted_type" != ext4 ]; then
    echo "[external-recording] Unexpected filesystem at $target; leaving it unchanged." >&2
    return 0
  fi
  # New logger routes need write access to the root. Avoid a recursive walk of
  # all old recordings at every boot; existing child ownership is not changed.
  if ! timeout -k 2s 5s sudo -n chown 1000:1000 -- "$target"; then
    echo "[external-recording] Could not set recording-root owner; check permissions." >&2
    return 0
  fi
  echo "[external-recording] Ready: $device -> $target"
  return 0
}

# Storage setup must not abort openpilot startup, including callers using set -e.
_sp_mount_external_recording || true
unset -f _sp_mount_external_recording
