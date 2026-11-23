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


class Voice:
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

  def silence(self, now):
    if self.request is not None:
      self.request = None
      self.generation += 1
      notify(None, 0, 0, now)

  def revoke(self, now=0.):
    self.silence(now)
    self.matcher.reset()

  def update(self, sm, service, now, permitted):
    age = now-sm.recv_time[service]
    if not permitted or not sm.alive[service] or not sm.valid[service] or not 0 <= age < FRESH:
      self.revoke(now)
      return
    if not sm.updated[service]:
      return
    gps = sm[service]
    if not gps.hasFix:
      self.revoke(now)
      return
    site = self.matcher.fix(sm.recv_time[service], gps.latitude, gps.longitude, gps.speed, gps.bearingDeg,
                            gps.horizontalAccuracy, gps.bearingAccuracyDeg)
    if site is None:
      self.silence(now)
      return
    changed = self.request is None or self.request[0] != site
    if changed:
      self.generation += 1
    self.request = (site, self.generation, self.samples[FAMILY[site.kind]], sm.recv_time[service]+FRESH)
    if changed or now-self.last_hud >= .4:
      self.last_hud = now
      notify(site.kind, site.limit, max(0., self.matcher.distance), now)

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
