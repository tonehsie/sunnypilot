"""GPS producer and bounded, I/O-free optional audio mixer. Native alerts always win."""
import json
import math
import wave
from pathlib import Path

import numpy as np

from .hud import notify
from .matcher import FAMILY, Matcher

ROOT = Path(__file__).resolve().parent
RATE = 48000
GAP = RATE * 3 // 4
FRESH = 2.
GPS_SERVICES = ("gpsLocationExternal", "gpsLocation")


def gps_input(gps, service):
  """Return a usable fix (and whether course needs track validation), not an invented accuracy."""
  if not gps.hasFix:
    return None
  lat, lon, speed, hacc = gps.latitude, gps.longitude, gps.speed, gps.horizontalAccuracy
  if not (20 <= lat <= 27 and 117 <= lon <= 124 and 0 <= speed <= 60):
    return None
  modem = service == "gpsLocation" and str(gps.source) == "qcomdiag"
  missing = modem and hacc == 0
  if not (0 < hacc <= 20 or missing):
    return None
  heading, uncertainty = gps.bearingDeg, gps.bearingAccuracyDeg
  if speed >= 2:
    reported = -180 <= heading <= 360 and 0 <= uncertainty <= 35
    if missing and len(gps.vNED) == 3:
      north, east, _ = gps.vNED
      horizontal_speed = math.hypot(north, east)
      if 2 <= horizontal_speed <= 60 and abs(horizontal_speed-speed) <= max(2., speed*.30):
        heading = math.degrees(math.atan2(east, north)) % 360
      elif not reported:
        return None
    elif not reported:
      return None
    heading %= 360
  return (lat, lon, speed, heading, hacc, uncertainty), missing


class Chimes:
  def __init__(self, root=ROOT):
    self.matcher = Matcher(json.loads((root / "points.json").read_text()))
    self.samples = {}
    for family in set(FAMILY.values()):
      with wave.open(str(root / "voice" / (family + ".wav")), "rb") as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, RATE) or not 0 < f.getnframes() <= 5*RATE:
          raise ValueError("invalid TW audio asset")
        clip = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32)/32768
        if len(clip) != f.getnframes():
          raise ValueError("truncated TW audio asset")
        self.samples[family] = clip
    self.request = None  # main thread publishes one immutable snapshot per fix
    self.generation = 0
    self.playing = None  # callback-owned cursor
    self.cursor = 0
    self.last_hud = -math.inf
    self.service = None

  def silence(self, now):
    if self.request is not None:
      self.request = None
      self.generation += 1
      notify(None, 0, 0, now)

  def revoke(self, now=0.):
    self.silence(now)
    self.matcher.reset()
    self.service = None

  def update(self, sm, now, permitted):
    if not permitted:
      self.revoke(now)
      return
    # Select by live data, not a startup-only copy of UbloxAvailable. Never mix
    # the previous receiver's coordinates or course into a fresh approach.
    for service in GPS_SERVICES:
      if sm.alive[service] and sm.valid[service] and 0 <= now-sm.recv_time[service] < FRESH:
        inputs = gps_input(sm[service], service)
        if inputs is not None:
          break
    else:
      self.revoke(now)
      return
    if self.service != service:
      self.revoke(now)
      self.service = service
    if not sm.updated[service]:
      return
    args, missing = inputs
    site = self.matcher.fix(sm.recv_time[service], *args, modem_track=missing)
    if site is None:
      self.silence(now)
      return
    changed = self.request is None or self.request[0] != site
    if changed:
      self.generation += 1
    self.request = (site, self.generation, self.samples[FAMILY[site.kind]], sm.recv_time[service]+FRESH)
    if changed or now-self.last_hud >= .4:
      self.last_hud = now
      notify(site.kind, site.limit, self.matcher.distance, now)

  def render(self, out, allowed, now):
    req = self.request
    if not allowed or req is None or now >= req[3]:
      self.cursor = 0
      self.playing = None
      return
    _, generation, clip, _ = req
    if generation != self.playing:
      self.cursor = 0
      self.playing = generation
    pos = 0
    while pos < len(out):
      if self.cursor < len(clip):
        n = min(len(out)-pos, len(clip)-self.cursor)
        buf = out[pos:pos+n]
        buf *= .2  # only silence / the optional speed-limit chime can reach this path
        buf += clip[self.cursor:self.cursor+n]
      else:
        n = min(len(out)-pos, len(clip)+GAP-self.cursor)
      self.cursor = (self.cursor+n) % (len(clip)+GAP)
      pos += n
