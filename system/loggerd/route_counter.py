"""Keep route IDs newer than retained logs after a params reset.

Called once by manager_init, before managed recording/deletion processes start.
Only scan directory names; never open, rename, or delete recordings.
"""

import os
import queue
import re
import threading
from collections.abc import Iterable


_SEGMENT_NAME = re.compile(r"([0-9a-fA-F]{8})--[0-9a-fA-F]{10}--[0-9]+")
_UINT32_MAX = (1 << 32) - 1


def _max_existing_route_id(log_roots: Iterable[str]) -> int:
  maximum = -1
  for root in log_roots:
    try:
      entries = os.scandir(root)
    except FileNotFoundError:
      continue  # first boot with no recording directory yet
    with entries:
      for entry in entries:
        match = _SEGMENT_NAME.fullmatch(entry.name)
        if match is not None and entry.is_dir(follow_symlinks=False):
          maximum = max(maximum, int(match.group(1), 16))
  return maximum


def reconcile_route_count(params, log_roots: Iterable[str], logger, timeout: float = 2.0) -> bool:
  """Raise the typed INT RouteCount only; return whether it was corrected.

  The daemon worker only reads directory metadata. A timeout or scan error
  cannot cause a late parameter write: all parameter access stays here.
  Must not be invoked while a recording process is running.
  """
  roots = tuple(log_roots)
  result: queue.Queue[int | Exception] = queue.Queue(maxsize=1)

  def scan() -> None:
    try:
      result.put(_max_existing_route_id(roots))
    except Exception as exc:
      result.put(exc)

  try:
    threading.Thread(target=scan, name="route-counter-scan", daemon=True).start()
    maximum = result.get(timeout=timeout)
  except queue.Empty:
    logger.warning("route counter reconciliation skipped: directory scan timed out")
    return False
  except (RuntimeError, OSError) as exc:
    logger.warning(f"route counter reconciliation skipped: {exc}")
    return False
  if isinstance(maximum, Exception):
    logger.warning(f"route counter reconciliation skipped: {maximum}")
    return False
  if maximum < 0:
    return False
  required = maximum + 1
  # logger_get_identifier stores cnt+1 in a uint32; do not write a wraparound.
  if required >= _UINT32_MAX:
    logger.warning("route counter reconciliation skipped: uint32 counter exhausted")
    return False

  try:
    current = params.get("RouteCount")
    current = 0 if current is None else current
    if type(current) is not int or not 0 <= current < _UINT32_MAX:
      logger.warning("route counter reconciliation skipped: invalid existing RouteCount")
      return False
    if current >= required:
      return False
    # release-tizi Params types RouteCount as INT, not a string.
    params.put("RouteCount", required, block=True)
    if params.get("RouteCount") != required:
      logger.warning("route counter reconciliation failed: parameter readback mismatch")
      return False
  except Exception as exc:
    logger.warning(f"route counter reconciliation failed: {exc}")
    return False

  logger.warning(f"route counter raised {current} -> {required}; retained max ID={maximum:08x}")
  return True
