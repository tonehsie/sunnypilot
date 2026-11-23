"""Offline approach tracker. No car-control or network dependencies."""
import math
from collections import defaultdict
from typing import NamedTuple

FAMILY = {"mobile": "mobile", "sign": "sign", **dict.fromkeys(("fixed", "section", "combined", "red", "tech", "listed"), "fixed")}
RANK = {"mobile": 0, "sign": 1, "fixed": 2}
PASS_M = 10.


class Site(NamedTuple):
  lat: float
  lon: float
  kind: str
  headings: tuple
  limit: int


def offset(lat, lon, a, b):
  return (b - lon) * 111320 * math.cos(math.radians(lat)), (a - lat) * 111320


def angle(a, b):
  return abs((a - b + 180) % 360 - 180)


def relative(lat, lon, site, course):
  x, y = offset(lat, lon, site.lat, site.lon)
  s, c = math.sin(math.radians(course)), math.cos(math.radians(course))
  return x*s + y*c, abs(x*c - y*s)


class Matcher:
  def __init__(self, rows):
    self.grid = defaultdict(list)
    for a, b, kind, dirs, limit, _ in rows:
      if kind not in FAMILY or not (20 <= a <= 27 and 117 <= b <= 124):
        raise ValueError("invalid point")
      site = Site(a, b, kind, tuple(dirs), limit)
      self.grid[int(a*100), int(b*100)].append(site)
    self.passed = {}
    self.reset()

  def reset(self):
    self.active = None
    self.course = 0.
    self.distance = 0.
    self.last = self.origin = None

  def fix(self, now, lat, lon, speed, bearing, accuracy, bearing_accuracy):
    moving = speed >= 2
    if not (all(map(math.isfinite, (now, lat, lon, speed, accuracy))) and 20 <= lat <= 27 and 117 <= lon <= 124
            and 0 <= speed <= 60 and 0 < accuracy <= 20 and
            (not moving or (0 <= bearing < 360 and 0 <= bearing_accuracy <= 35))):
      self.reset()
      return None
    old = self.last
    if old is None or not 0 < now-old[0] <= 2:
      self.reset()
      self.last = self.origin = (now, lat, lon)
      return None
    self.last = (now, lat, lon)
    if math.hypot(*offset(old[1], old[2], lat, lon)) > speed*(now-old[0])*2 + max(15, accuracy):
      self.reset()
      return None
    dx, dy = offset(self.origin[1], self.origin[2], lat, lon)
    progress = math.hypot(dx, dy) >= max(3, accuracy*.6)
    if progress:
      self.origin = self.last
      if moving and angle(math.degrees(math.atan2(dx,dy)), bearing) > 45:
        self.active = None
        return None

    self.passed = {site:p for site,p in self.passed.items() if now-p[0] < 90}
    if self.active is not None:
      # Keep one approach axis near the point: GPS bearing can wander at the stop line.
      ahead, side = relative(lat, lon, self.active, self.course)
      if ahead <= -PASS_M + 1e-6:
        self.passed[self.active] = (now, self.course)
        self.active = None
      elif side > 45 or (moving and angle(bearing, self.course) > 60):
        self.active = None  # turned off the road
      else:
        self.distance = ahead
    if not moving or not progress:
      return self.active
    # Existing target stays latched through zero; only a higher-priority candidate
    # may interrupt it. Equal-priority candidates are nearest first on acquisition.
    best = None
    reach = max(110., min(480., speed*14))
    for i in range(int(lat*100)-1, int(lat*100)+2):
      for j in range(int(lon*100)-1, int(lon*100)+2):
        for site in self.grid.get((i,j), ()):
          rank = RANK[FAMILY[site.kind]]
          if self.active is not None and rank >= RANK[FAMILY[self.active.kind]]:
            continue
          if site in self.passed and angle(self.passed[site][1],bearing) < 75:
            continue
          if site.headings and all(angle(bearing,h)>55 for h in site.headings):
            continue
          ahead, side = relative(lat,lon,site,bearing)
          if 5 <= ahead <= reach and side <= (26 if site.headings else 15):
            key = (rank, ahead, side)
            if best is None or key < best[0]:
              best = (key, site)
    if best:
      self.active, self.distance, self.course = best[1], best[0][1], bearing
    return self.active
