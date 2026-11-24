"""Offline approach tracker. No car-control or network dependencies."""
import math
from collections import defaultdict
from typing import NamedTuple

FAMILY = {"mobile": "mobile", "sign": "sign", **dict.fromkeys(("fixed", "section", "combined", "red", "tech", "listed"), "fixed")}
RANK = {"mobile": 0, "sign": 1, "fixed": 2}
CAMERA_BEFORE = {"ordinary": 300., "express": 600., "freeway": 800.}
CAMERA_AFTER = 10.
SIGN_BEFORE, SIGN_AFTER = 10., 300.
EPS = 1e-6


class Site(NamedTuple):
  lat: float
  lon: float
  kind: str
  headings: tuple
  limit: int
  road: str


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
    for a, b, kind, dirs, limit, _, *road in rows:
      if kind not in FAMILY or not (20 <= a <= 27 and 117 <= b <= 124):
        raise ValueError("invalid point")
      road_class = road[0] if road else "ordinary"
      if road_class not in CAMERA_BEFORE:
        raise ValueError("invalid road class")
      site = Site(a, b, kind, tuple(dirs), limit, road_class)
      self.grid[int(a*100), int(b*100)].append(site)
    self.passed = {}
    self.reset()

  def reset(self):
    self.active = None
    self.encounters = {}  # independent timers/distances, even for temporarily suppressed bells
    self.trusted_course = None
    self.course = 0.
    self.distance = 0.
    self.last = self.origin = None

  def fix(self, now, lat, lon, speed, bearing, accuracy, bearing_accuracy, *, modem_track=False):
    moving = speed >= 2
    if not (all(map(math.isfinite, (now, lat, lon, speed, accuracy))) and 20 <= lat <= 27 and 117 <= lon <= 124
            and 0 <= speed <= 60 and (0 < accuracy <= 20 or (modem_track and accuracy == 0)) and
            (not moving or (0 <= bearing < 360 and (modem_track or 0 <= bearing_accuracy <= 35)))):
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
    origin = self.origin
    dx, dy = offset(origin[1], origin[2], lat, lon)
    # Unknown modem precision is not a perfect fix. Require actual displacement
    # corroborating the modem's course before acquiring a point.
    progress = math.hypot(dx, dy) >= (12. if modem_track else max(3, accuracy*.6))
    if progress:
      self.origin = self.last
      if moving and angle(math.degrees(math.atan2(dx,dy)), bearing) > 45:
        self.reset()
        return None
      if moving:
        self.trusted_course = bearing

    self.passed = {site:p for site,p in self.passed.items() if now-p[0] < 90}
    for site, (axis, dist) in list(self.encounters.items()):
      is_sign = site.kind == "sign"
      if is_sign and dist <= 0:
        # Beyond a sign, integrate forward movement along the local road rather
        # than a straight chord from the sign; stationary jitter isn't distance.
        if moving and angle(bearing, axis) > 55:
          del self.encounters[site]
          continue
        x, y = offset(old[1], old[2], lat, lon)
        course = bearing if moving else axis
        rad = math.radians(course)
        if speed >= .5:
          dist -= x*math.sin(rad) + y*math.cos(rad)
        axis = course
      else:
        dist, side = relative(lat, lon, site, axis)
        if side > 45 or (moving and angle(bearing,axis) > 60):
          del self.encounters[site]
          continue
      if dist <= -(SIGN_AFTER if is_sign else CAMERA_AFTER) + EPS:
        self.passed[site] = (now, axis)
        del self.encounters[site]
      else:
        self.encounters[site] = (axis, dist)

    # Once course is corroborated, check every fresh moving fix, not just
    # every 12 m: the entire pre-sign window can lie between two 1 Hz fixes.
    if moving and self.trusted_course is not None and angle(bearing,self.trusted_course) <= 45:
      radius = max(CAMERA_BEFORE.values()) + 45
      dlat = radius/111320
      dlon = dlat/math.cos(math.radians(lat))
      for i in range(int((lat-dlat)*100), int((lat+dlat)*100)+1):
        for j in range(int((lon-dlon)*100), int((lon+dlon)*100)+1):
          for site in self.grid.get((i,j), ()):
            if site in self.encounters or (site in self.passed and angle(self.passed[site][1],bearing) < 75):
              continue
            if site.headings and all(angle(bearing,h)>55 for h in site.headings):
              continue
            ahead, side = relative(lat,lon,site,bearing)
            width = 26 if site.headings else 15
            if side > width:
              continue
            if site.kind == "sign":
              if ahead > SIGN_BEFORE + EPS:
                continue
              if ahead < 0:
                prev, prev_side = relative(old[1],old[2],site,bearing)
                if prev < 0 or prev_side > width or ahead <= -SIGN_AFTER + EPS:
                  continue  # no observed crossing; don't arm a sign already behind us
            elif not 0 <= ahead <= CAMERA_BEFORE[site.road] + EPS:
              continue
            self.encounters[site] = (bearing, ahead)
    # Keep the current point among equal ranks. Silenced encounters still advance.
    self.active = min(self.encounters, key=lambda s:(RANK[FAMILY[s.kind]], s != self.active,
                      abs(self.encounters[s][1])), default=None)
    if self.active is not None:
      self.course, self.distance = self.encounters[self.active]
    return self.active
